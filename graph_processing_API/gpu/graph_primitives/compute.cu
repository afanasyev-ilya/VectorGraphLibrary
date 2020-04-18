#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight, typename ComputeOperation>
void GraphPrimitivesGPU::compute(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                 FrontierGPU &_frontier,
                                 ComputeOperation &&compute_op)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    long long *vertex_pointers = outgoing_ptrs;

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

template<typename ComputeOperation>
void __global__ compute_kernel_all_active(const int _frontier_size,
                                          const long long *_vertex_pointers,
                                          ComputeOperation compute_op)
{
    const int src_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(src_id < _frontier_size)
    {
        int connections_count = _vertex_pointers[src_id + 1] - _vertex_pointers[src_id];
        compute_op(src_id, connections_count);
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
            compute_op(src_id, connections_count);
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
        compute_op(src_id, connections_count);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////