#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename _TVertexValue, typename _TEdgeWeight, typename ReduceOperation>
_T GraphPrimitivesGPU::reduce(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                              FrontierGPU &_frontier,
                              ReduceOperation &&reduce_op,
                              REDUCE_TYPE _reduce_type)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);
    long long *vertex_pointers = outgoing_ptrs;

    _T *reduce_tmp_array;
    MemoryAPI::allocate_device_array(&reduce_tmp_array, vertices_count);
    cudaMemset(reduce_tmp_array, 0, sizeof(_T) * vertices_count);

    _T reduce_result = 0;
    if(_frontier.type == ALL_ACTIVE_FRONTIER)
    {
        SAFE_KERNEL_CALL((reduce_kernel_all_active <<< (vertices_count - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >>>
                  (reduce_tmp_array, vertices_count, vertex_pointers, reduce_op)));
        if(_reduce_type == REDUCE_SUM)
        {
            reduce_result = thrust::reduce(thrust::device, reduce_tmp_array, reduce_tmp_array + vertices_count);
        }
    }
    else if(_frontier.type == DENSE_FRONTIER)
    {
        SAFE_KERNEL_CALL((reduce_kernel_dense <<< (vertices_count - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >>>
                  (reduce_tmp_array, _frontier.flags, vertices_count, vertex_pointers, reduce_op)));
        if(_reduce_type == REDUCE_SUM)
        {
            reduce_result = thrust::reduce(thrust::device, reduce_tmp_array, reduce_tmp_array + vertices_count);
        }
    }
    else if(_frontier.type == SPARSE_FRONTIER)
    {
        int frontier_size = _frontier.size();
        SAFE_KERNEL_CALL((reduce_kernel_sparse <<< (frontier_size - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >>>
                   (reduce_tmp_array, _frontier.ids, frontier_size, vertex_pointers, reduce_op)));
        if(_reduce_type == REDUCE_SUM)
        {
            reduce_result = thrust::reduce(thrust::device, reduce_tmp_array, reduce_tmp_array + frontier_size);
        }
    }

    MemoryAPI::free_device_array(reduce_tmp_array);
    cudaDeviceSynchronize();

    return reduce_result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template<typename ReduceOperation>
void __global__ reduce_kernel_all_active(int *_reduce_tmp_array,
                                         const int _frontier_size,
                                         const long long *_vertex_pointers,
                                         ReduceOperation reduce_op)
{
    const int src_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(src_id < _frontier_size)
    {
        int connections_count = _vertex_pointers[src_id + 1] - _vertex_pointers[src_id];
        _reduce_tmp_array[src_id] = reduce_op(src_id, connections_count);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename ReduceOperation>
void __global__ reduce_kernel_dense(int *_reduce_tmp_array,
                                    const int *_frontier_flags,
                                    const int _frontier_size,
                                    const long long *_vertex_pointers,
                                    ReduceOperation reduce_op)
{
    const int src_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(src_id < _frontier_size)
    {
        if(_frontier_flags[src_id] > 0)
        {
            int connections_count = _vertex_pointers[src_id + 1] - _vertex_pointers[src_id];
            _reduce_tmp_array[src_id] = reduce_op(src_id, connections_count);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename ReduceOperation>
void __global__ reduce_kernel_sparse(int *_reduce_tmp_array,
                                     const int *_frontier_ids,
                                     const int _frontier_size,
                                     const long long *_vertex_pointers,
                                     ReduceOperation reduce_op)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < _frontier_size)
    {
        int src_id = _frontier_ids[idx];
        int connections_count = _vertex_pointers[src_id + 1] - _vertex_pointers[src_id];
        _reduce_tmp_array[idx] = reduce_op(src_id, connections_count);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////