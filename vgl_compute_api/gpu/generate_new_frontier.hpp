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

template<typename GraphContainer>
void __global__ split_frontier(GraphContainer _graph,
                               int *_work_buffer,
                               int *_frontier_ids,
                               const int _size,
                               int *_vector_engine_threshold,
                               int *_vector_core_threshold)
{
    register const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < _size)
    {
        _frontier_ids[idx] = _work_buffer[idx];
        int current_id = _work_buffer[idx];
        int next_id = _work_buffer[idx + 1];

        int current_size = _graph.get_connections_count(current_id);
        int next_size = 0;
        if(idx < (_size - 1))
        {
            next_size = _graph.get_connections_count(next_id);
        }

        if((current_size > VECTOR_ENGINE_THRESHOLD_VALUE) && (next_size <= VECTOR_ENGINE_THRESHOLD_VALUE))
        {
            *_vector_engine_threshold = idx + 1;
        }
        if((current_size > VECTOR_CORE_THRESHOLD_VALUE) && (next_size <= VECTOR_CORE_THRESHOLD_VALUE))
        {
            *_vector_core_threshold = idx + 1;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FilterCondition>
void GraphAbstractionsGPU::generate_new_frontier_worker(VectorCSRGraph &_graph,
                                                        FrontierVectorCSR &_frontier,
                                                        FilterCondition &&filter_cond)
{
    Timer tm;
    tm.start();
    _frontier.set_direction(current_traversal_direction);
    int vertices_count = _graph.get_vertices_count();
    LOAD_FRONTIER_DATA(_frontier);

    int *ve_threshold;
    int *vc_threshold;
    MemoryAPI::allocate_array(&ve_threshold, 1);
    MemoryAPI::allocate_array(&vc_threshold, 1);
    ve_threshold[0] = 0;
    vc_threshold[0] = 0;

    // generate frontier flags
    dim3 grid((vertices_count - 1) / BLOCK_SIZE + 1);
    dim3 block(BLOCK_SIZE);
    SAFE_KERNEL_CALL((set_frontier_flags<<<grid, block>>>(_graph, frontier_ids, frontier_flags,
            vertices_count, filter_cond)));

    auto copy_if_cond = [frontier_flags, frontier_ids] __host__ __device__ (int _src_id)->bool {
        if(_src_id >= 0)
            return true;
        else
            return false;
    };

    _frontier.sparsity_type = SPARSE_FRONTIER;
    _frontier.vector_engine_part_type = SPARSE_FRONTIER;
    _frontier.vector_core_part_type = SPARSE_FRONTIER;
    _frontier.collective_part_type = SPARSE_FRONTIER;

    int copied_elements = ParallelPrimitives::copy_if_data(copy_if_cond, frontier_ids, frontier_work_buffer,
                                                           vertices_count, frontier_work_buffer, vertices_count);
    _frontier.size = copied_elements;

    dim3 split_grid((copied_elements - 1) / BLOCK_SIZE + 1);
    SAFE_KERNEL_CALL((split_frontier<<<split_grid, block>>>(_graph, frontier_work_buffer, frontier_ids, copied_elements,
                        ve_threshold, vc_threshold)));
    _frontier.vector_engine_part_size = ve_threshold[0];
    _frontier.vector_core_part_size = vc_threshold[0] - ve_threshold[0];
    _frontier.collective_part_size = copied_elements - vc_threshold[0];

    if (_frontier.size == _graph.get_vertices_count())
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

    MemoryAPI::free_array(ve_threshold);
    MemoryAPI::free_array(vc_threshold);

    tm.end();
    performance_stats.update_gnf_time(tm);
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_time_and_bandwidth_stats("GNF", vertices_count, 4.0*sizeof(int));
    #endif
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

    // generate frontier IDS
    int *new_end = thrust::remove_if(thrust::device, frontier_ids, frontier_ids + vertices_count, is_not_active()); // 2*|V|
    _frontier.size = new_end - _frontier.ids;

    if (_frontier.size == _graph.get_vertices_count())
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
void GraphAbstractionsGPU::generate_new_frontier_worker(CSR_VG_Graph &_graph,
                                                        FrontierCSR_VG &_frontier,
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

    auto filter_vertex_group = [frontier_flags] __host__ __device__ (int _src_id)->bool {
        return frontier_flags[_src_id];
    };
    _frontier.copy_vertex_group_info_from_graph_cond(filter_vertex_group);

    int copy_pos = 0;
    for(int i = 0; i < CSR_VERTEX_GROUPS_NUM; i++)
    {
        cudaMemcpy(frontier_ids + copy_pos, _frontier.vertex_groups[i].ids, _frontier.vertex_groups[i].size * sizeof(int),
                   cudaMemcpyDeviceToDevice);
        copy_pos += _frontier.vertex_groups[i].size;
    }
    _frontier.size = _frontier.get_size_of_vertex_groups();

    if (_frontier.size == _graph.get_vertices_count())
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
