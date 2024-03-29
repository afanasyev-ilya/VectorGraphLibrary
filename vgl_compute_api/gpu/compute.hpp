#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename ComputeOperation, typename GraphContainer>
void __global__ compute_kernel_all_active(GraphContainer _graph,
                                          const int _frontier_size,
                                          ComputeOperation compute_op)
{
    const int src_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(src_id < _frontier_size)
    {
        int connections_count = _graph.get_connections_count(src_id);
        int vector_index = lane_id();
        compute_op(src_id, connections_count, vector_index);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename ComputeOperation, typename GraphContainer>
void __global__ compute_kernel_dense(GraphContainer _graph,
                                     int *_frontier_flags,
                                     const int _frontier_size,
                                     ComputeOperation compute_op)
{
    const int src_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(src_id < _frontier_size)
    {
        if(_frontier_flags[src_id] > 0)
        {
            int connections_count = _graph.get_connections_count(src_id);
            int vector_index = lane_id();
            compute_op(src_id, connections_count, vector_index);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename ComputeOperation, typename GraphContainer>
void __global__ compute_kernel_sparse(GraphContainer _graph,
                                      int *_frontier_ids,
                                      const int _frontier_size,
                                      ComputeOperation compute_op)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < _frontier_size)
    {
        int src_id = _frontier_ids[idx];
        int connections_count = _graph.get_connections_count(src_id);
        int vector_index = lane_id();
        compute_op(src_id, connections_count, vector_index);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ComputeOperation, typename GraphContainer, typename FrontierContainer>
void GraphAbstractionsGPU::compute_worker(GraphContainer &_graph,
                                          FrontierContainer &_frontier,
                                          ComputeOperation &&compute_op)
{
    int vertices_count = _graph.get_vertices_count();
    LOAD_FRONTIER_DATA(_frontier);

    if(_frontier.get_sparsity_type() == ALL_ACTIVE_FRONTIER)
    {
        SAFE_KERNEL_CALL((compute_kernel_all_active <<< (vertices_count - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >>>
        (_graph, vertices_count, compute_op)));
    }
    else if(_frontier.get_sparsity_type() == DENSE_FRONTIER)
    {
        SAFE_KERNEL_CALL((compute_kernel_dense <<< (vertices_count - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >>>
        (_graph, frontier_flags, vertices_count, compute_op)));
    }
    else if(_frontier.get_sparsity_type() == SPARSE_FRONTIER)
    {
        SAFE_KERNEL_CALL((compute_kernel_sparse <<< (frontier_size - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >>>
        (_graph, frontier_ids, frontier_size, compute_op)));
    }
    cudaDeviceSynchronize();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
