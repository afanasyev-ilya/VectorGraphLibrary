#pragma once

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

template <typename Condition>
void __global__ copy_frontier_ids_kernel(int *_frontier_ids,
                                         int *_frontier_flags,
                                         long long *_vertex_pointers,
                                         const int _vertices_count,
                                         Condition cond)
{
    register const int src_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(src_id < _vertices_count)
    {
        int connections_count = _vertex_pointers[src_id + 1] - _vertex_pointers[src_id];
        if(cond(src_id, connections_count) == true)
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

template <typename Condition>
void GraphAbstractionsGPU::generate_new_frontier(VectCSRGraph &_graph,
                                                 FrontierGPU &_frontier,
                                                 Condition &&cond)
{
    Timer tm;
    tm.start();
    _frontier.set_direction(current_traversal_direction);
    _frontier.type = SPARSE_FRONTIER; // TODO

    UndirectedCSRGraph *graph_ptr = _graph.get_direction_graph_ptr(current_traversal_direction);
    LOAD_UNDIRECTED_CSR_GRAPH_DATA((*graph_ptr));

    SAFE_KERNEL_CALL((copy_frontier_ids_kernel<<<(vertices_count - 1)/BLOCK_SIZE + 1, BLOCK_SIZE>>>(_frontier.ids, _frontier.flags,
                                                                                                    vertex_pointers, vertices_count, cond)));

    int *new_end = thrust::remove_if(thrust::device, _frontier.ids, _frontier.ids + vertices_count, is_not_active());
    _frontier.current_size = new_end - _frontier.ids;
    cudaDeviceSynchronize();
    tm.end();
    performance_stats.update_gnf_time(tm);
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_bandwidth_stats("GNF", _frontier.size(), 4.0*sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////