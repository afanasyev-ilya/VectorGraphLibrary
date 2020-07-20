#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#if __CUDA_ARCH__ >= 700
#define FULL_MASK 0xffffffff
#endif

template <typename _T>
static __device__ __forceinline__ _T shfl_down( _T r, int offset )
{
    #if __CUDA_ARCH__ >= 700
    return __shfl_down_sync(FULL_MASK, r, offset );
    #elif __CUDA_ARCH__ >= 300
    return __shfl_down( r, offset );
    #else
    return 0.0f;
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*#if __CUDA_ARCH__ < 500
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
            (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                                             __longlong_as_double(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif*/

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
__inline__ __device__ _T warp_reduce_sum(_T val)
{
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += shfl_down(val, offset);
    return val;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
__inline__ __device__ _T block_reduce_sum(_T val)
{
    static __shared__ _T shared[32]; // Shared mem for 32 partial sums
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warp_reduce_sum(val);     // Each warp performs partial reduction

    if (lane==0)
        shared[wid]=val; // Write reduced value to shared memory

    __syncthreads();              // Wait for all partial reductions

    //read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    if(wid == 0)
        val = warp_reduce_sum(val); //Final reduce within first warp

    return val;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename ReduceOperation>
__global__ void reduce_kernel_sparse(const int *_frontier_ids,
                                     const int _frontier_size,
                                     const long long *_vertex_pointers,
                                     ReduceOperation reduce_op,
                                     _T* out)
{
    const int frontier_pos = blockIdx.x * blockDim.x + threadIdx.x;

    if(frontier_pos < _frontier_size)
    {
        int src_id = _frontier_ids[frontier_pos];
        int connections_count = _vertex_pointers[src_id + 1] - _vertex_pointers[src_id];

        _T sum = reduce_op(src_id, frontier_pos, connections_count);

        sum = block_reduce_sum(sum);
        if (threadIdx.x == 0)
            atomicAdd(out, sum);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename ReduceOperation>
__global__ void reduce_kernel_all_active(const int _size,
                                         const long long *_vertex_pointers,
                                         ReduceOperation reduce_op,
                                         _T* _result)
{
    const int src_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(src_id < _size)
    {
        int connections_count = _vertex_pointers[src_id + 1] - _vertex_pointers[src_id];
        _T sum = reduce_op(src_id, src_id, connections_count);

        sum = block_reduce_sum(sum);
        if (threadIdx.x == 0)
            atomicAdd(_result, sum);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename _TVertexValue, typename _TEdgeWeight, typename ReduceOperation>
_T GraphPrimitivesGPU::reduce(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                              FrontierGPU &_frontier,
                              ReduceOperation &&reduce_op,
                              REDUCE_TYPE _reduce_type)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);
    long long *vertex_pointers = outgoing_ptrs;

    _T *managed_reduced_result;
    MemoryAPI::allocate_managed_array(&managed_reduced_result, 1);

    if(_frontier.type == ALL_ACTIVE_FRONTIER)
    {
        SAFE_KERNEL_CALL((reduce_kernel_all_active<<< (vertices_count - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >>>(vertices_count, vertex_pointers, reduce_op, managed_reduced_result)));
    }
    else if(_frontier.type == DENSE_FRONTIER)
    {
        throw "Error: dense frontier in reduce is not supported";
    }
    else if(_frontier.type == SPARSE_FRONTIER)
    {
        int frontier_size = _frontier.size();
        SAFE_KERNEL_CALL((reduce_kernel_sparse<<< (frontier_size - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >>>(_frontier.ids, frontier_size, vertex_pointers, reduce_op, managed_reduced_result)));
    }

    cudaDeviceSynchronize();
    _T reduce_result = managed_reduced_result[0];

    //MemoryAPI::free_array(managed_reduced_result);

    return reduce_result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
