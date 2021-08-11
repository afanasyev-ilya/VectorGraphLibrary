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
void __global__ copy_frontier_ids_kernel(GraphContainer *_graph,
                                         int *_frontier_ids,
                                         int *_frontier_flags,
                                         const int _vertices_count,
                                         FilterCondition filter_cond)
{
    register const int src_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(src_id < _vertices_count)
    {
        int connections_count = _graph->get_connections_count(src_id);
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

template <typename FilterCondition, typename GraphContainer, typename FrontierContainer>
void GraphAbstractionsGPU::generate_new_frontier_worker(GraphContainer &_graph,
                                                        FrontierContainer &_frontier,
                                                        FilterCondition &&filter_cond)
{
    throw "Error in GraphAbstractionsGPU::generate_new_frontier_worker : not implemented yet";
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FilterCondition, typename GraphContainer>
void GraphAbstractionsGPU::generate_new_frontier_worker(GraphContainer &_graph,
                                                        FrontierGeneral &_frontier,
                                                        FilterCondition &&filter_cond)
{
    Timer tm;
    tm.start();
    _frontier.set_direction(current_traversal_direction);
    _frontier.sparsity_type = SPARSE_FRONTIER; // TODO

    int vertices_count = _graph.get_vertices_count();
    int *frontier_ids = _frontier.get_ids();
    int *frontier_flags = _frontier.get_flags();

    // generate frontier flags
    SAFE_KERNEL_CALL((copy_frontier_ids_kernel<<<(vertices_count - 1) / BLOCK_SIZE +
                                                 1, BLOCK_SIZE>>>(&_graph, frontier_ids, frontier_flags,
                                                 vertices_count, filter_cond))); // 2*|V|

    // generate frontier IDS
    int *new_end = thrust::remove_if(thrust::device, frontier_ids, frontier_ids + vertices_count, is_not_active()); // 2*|V|

    // calculate frontier size
    _frontier.size = new_end - _frontier.ids;
    if (_frontier.get_size() == _graph.get_vertices_count())
    {
        _frontier.sparsity_type = ALL_ACTIVE_FRONTIER;
        _frontier.neighbours_count = _graph.get_edges_count();
    }
    else
    {
        _frontier.sparsity_type = SPARSE_FRONTIER;
        auto reduce_connections = [] __VGL_REDUCE_INT_ARGS__
        {
            return connections_count;
        };
        reduce_worker_sum(_graph, _frontier, reduce_connections, _frontier.neighbours_count);
        //_frontier.neighbours_count = _graph.get_edges_count(); // TODO replace with correct
    }

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
