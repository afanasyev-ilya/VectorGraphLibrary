#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename ComputeOperation>
void __global__ compute_kernel_all_active(const int _frontier_size,
                                          const long long *_vertex_pointers,
                                          ComputeOperation compute_op)
{
    const int src_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(src_id < _frontier_size)
    {
        int connections_count = _vertex_pointers[src_id + 1] - _vertex_pointers[src_id];
        int vector_index = cub::LaneId();
        compute_op(src_id, connections_count, vector_index);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename ComputeOperation>
void __global__ compute_kernel_dense(const int *_frontier_flags,
                                     const int _frontier_size,
                                     const long long *_vertex_pointers,
                                     ComputeOperation compute_op)
{
    const int src_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(src_id < _frontier_size)
    {
        if(_frontier_flags[src_id] > 0)
        {
            int connections_count = _vertex_pointers[src_id + 1] - _vertex_pointers[src_id];
            int vector_index = cub::LaneId();
            compute_op(src_id, connections_count, vector_index);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename ComputeOperation>
void __global__ compute_kernel_sparse(const int *_frontier_ids,
                                      const int _frontier_size,
                                      const long long *_vertex_pointers,
                                      ComputeOperation compute_op)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < _frontier_size)
    {
        int src_id = _frontier_ids[idx];
        int connections_count = _vertex_pointers[src_id + 1] - _vertex_pointers[src_id];
        int vector_index = cub::LaneId();
        compute_op(src_id, connections_count, vector_index);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ComputeOperation>
void GraphAbstractionsGPU::compute_worker(UndirectedCSRGraph &_graph,
                                          FrontierGPU &_frontier,
                                          ComputeOperation &&compute_op)
{
    LOAD_UNDIRECTED_CSR_GRAPH_DATA(_graph);

    if(_frontier.type == ALL_ACTIVE_FRONTIER)
    {
        SAFE_KERNEL_CALL((compute_kernel_all_active <<< (vertices_count - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >>>
                (vertices_count, vertex_pointers, compute_op)));
    }
    else if(_frontier.type == DENSE_FRONTIER)
    {
        SAFE_KERNEL_CALL((compute_kernel_dense <<< (vertices_count - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >>>
                (_frontier.flags, vertices_count, vertex_pointers, compute_op)));
    }
    else if(_frontier.type == SPARSE_FRONTIER)
    {
        int frontier_size = _frontier.size();
        SAFE_KERNEL_CALL((compute_kernel_sparse <<< (frontier_size - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >>>
                (_frontier.ids, frontier_size, vertex_pointers, compute_op)));
    }
    cudaDeviceSynchronize();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ComputeOperation>
void GraphAbstractionsGPU::compute(VectCSRGraph &_graph,
                                   FrontierGPU &_frontier,
                                   ComputeOperation &&compute_op)
{
    Timer tm;
    tm.start();

    if(_frontier.get_direction() != current_traversal_direction)
    {
        throw "Error in GraphAbstractionsGPU::compute : wrong frontier direction";
    }

    UndirectedCSRGraph *current_direction_graph;
    if(current_traversal_direction == SCATTER)
    {
        current_direction_graph = _graph.get_outgoing_graph_ptr();
    }
    else if(current_traversal_direction == GATHER)
    {
        current_direction_graph = _graph.get_incoming_graph_ptr();
    }

    compute_worker(*current_direction_graph, _frontier, compute_op);

    tm.end();
    performance_stats.update_compute_time(tm);
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_time_and_bandwidth_stats("Compute", _frontier.size(), COMPUTE_INT_ELEMENTS*sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

