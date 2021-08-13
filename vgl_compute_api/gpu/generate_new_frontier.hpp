////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct is_active
{
    __device__
    bool operator()(const int x)
    {
        return x != -1;
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct is_not_active
{
    __device__
    bool operator()(const int x)
    {
        return x == -1;
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename FilterCondition, typename GraphContainer>
void __global__ set_frontier_flags(GraphContainer _graph,
                                   int *_frontier_ids,
                                   int *_frontier_flags,
                                   const int _vertices_count,
                                   FilterCondition filter_cond)
{
    register const int src_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(src_id < _vertices_count)
    {
        int connections_count = _graph.get_connections_count(src_id);
        if(filter_cond(src_id, connections_count) == true)
        {
            _frontier_ids[src_id] = src_id;
            _frontier_flags[src_id] = IN_FRONTIER_FLAG;
        }
        else
        {
            _frontier_ids[src_id] = -1;
            _frontier_flags[src_id] = NOT_IN_FRONTIER_FLAG;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FilterCondition>
void GraphAbstractionsGPU::generate_new_frontier_worker(VectorCSRGraph &_graph,
                                                        FrontierVectorCSR &_frontier,
                                                        FilterCondition &&filter_cond)
{
    throw "Error in GraphAbstractionsGPU::generate_new_frontier_worker : not implemented yet";
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FilterCondition>
void GraphAbstractionsGPU::generate_new_frontier_worker(CSRGraph &_graph,
                                                        FrontierCSR &_frontier,
                                                        FilterCondition &&filter_cond)
{
    Timer tm;
    tm.start();
    _frontier.set_direction(current_traversal_direction);
    _frontier.sparsity_type = SPARSE_FRONTIER; // TODO

    int vertices_count = _graph.get_vertices_count();
    LOAD_FRONTIER_DATA(_frontier);

    // generate frontier flags
    SAFE_KERNEL_CALL((set_frontier_flags<<<(vertices_count - 1) / BLOCK_SIZE +
                                                 1, BLOCK_SIZE>>>(_graph, frontier_ids, frontier_flags,
                                                 vertices_count, filter_cond))); // 2*|V|

    #ifdef __USE_CSR_VERTEX_GROUPS__
    auto filter_vertex_group = [frontier_flags] __host__ __device__ (int _src_id)->bool {
        return frontier_flags[_src_id];
    };
    _frontier.copy_vertex_group_info_from_graph_cond(filter_vertex_group);

    int copy_pos = 0;
    cudaMemcpy(frontier_ids + copy_pos, _frontier.large_degree.ids, _frontier.large_degree.size, cudaMemcpyDeviceToDevice);
    copy_pos += _frontier.large_degree.size;
    cudaMemcpy(frontier_ids + copy_pos, _frontier.degree_32_1024.ids, _frontier.degree_32_1024.size, cudaMemcpyDeviceToDevice);
    copy_pos += _frontier.degree_32_1024.size;
    cudaMemcpy(frontier_ids + copy_pos, _frontier.degree_16_32.ids, _frontier.degree_16_32.size, cudaMemcpyDeviceToDevice);
    copy_pos += _frontier.degree_16_32.size;
    cudaMemcpy(frontier_ids + copy_pos, _frontier.degree_8_16.ids, _frontier.degree_8_16.size, cudaMemcpyDeviceToDevice);
    copy_pos += _frontier.degree_8_16.size;
    cudaMemcpy(frontier_ids + copy_pos, _frontier.degree_4_8.ids, _frontier.degree_4_8.size, cudaMemcpyDeviceToDevice);
    copy_pos += _frontier.degree_4_8.size;
    cudaMemcpy(frontier_ids + copy_pos, _frontier.degree_0_4.ids, _frontier.degree_0_4.size, cudaMemcpyDeviceToDevice);
    copy_pos += _frontier.degree_0_4.size;

    _frontier.size = _frontier.get_size_of_vertex_groups();
    #else
    // generate frontier IDS
    int *new_end = thrust::remove_if(thrust::device, frontier_ids, frontier_ids + vertices_count, is_not_active()); // 2*|V|
    _frontier.size = new_end - _frontier.ids;
    #endif

    if (_frontier.get_size() == _graph.get_vertices_count())
    {
        _frontier.sparsity_type = ALL_ACTIVE_FRONTIER;
        _frontier.neighbours_count = _graph.get_edges_count();
    }
    else
    {
        _frontier.sparsity_type = SPARSE_FRONTIER;
        auto reduce_connections = [] __VGL_REDUCE_INT_ARGS__ {
                return connections_count;
        };
        reduce_worker_sum(_graph, _frontier, reduce_connections, _frontier.neighbours_count);
    }

    cudaDeviceSynchronize();

    tm.end();
    performance_stats.update_gnf_time(tm);
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_time_and_bandwidth_stats("GNF", vertices_count, 4.0*sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FilterCondition>
void GraphAbstractionsGPU::generate_new_frontier_worker(EdgesListGraph &_graph,
                                                        FrontierEdgesList &_frontier,
                                                        FilterCondition &&filter_cond)
{
    Timer tm;
    tm.start();

    int vertices_count = _graph.get_vertices_count();
    LOAD_FRONTIER_DATA(_frontier);

    _frontier.set_direction(current_traversal_direction);
    _frontier.sparsity_type = ALL_ACTIVE_FRONTIER;

    // generate frontier flags
    SAFE_KERNEL_CALL((set_frontier_flags<<<(vertices_count - 1) / BLOCK_SIZE +
         1, BLOCK_SIZE>>>(_graph, frontier_ids, frontier_flags, vertices_count, filter_cond))); // 2*|V|

    _frontier.size = thrust::count_if(thrust::device, frontier_ids, frontier_ids + vertices_count, is_active());
    cudaDeviceSynchronize();

    tm.end();
    performance_stats.update_gnf_time(tm);
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_bandwidth_stats("GNF", vertices_count, 4.0*sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FilterCondition>
void GraphAbstractionsGPU::generate_new_frontier(VGL_Graph &_graph,
                                                 VGL_Frontier &_frontier,
                                                 FilterCondition &&filter_cond)
{
    common_generate_new_frontier(_graph, _frontier, filter_cond, this);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
