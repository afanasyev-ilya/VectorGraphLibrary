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

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
static __inline__ __device__ double atomicAdd(double *address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    if (val==0.0)
      return __longlong_as_double(old);
    do {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +__longlong_as_double(assumed)));
    }
    while (assumed != old);
    return __longlong_as_double(old);
}
#endif

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
    int lane =  lane_id();
    int wid =  warp_id();

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

template <typename _T, typename ReduceOperation, typename GraphContainer>
__global__ void reduce_kernel_all_active(GraphContainer _graph,
                                         const int _size,
                                         ReduceOperation reduce_op,
                                         _T* _result)
{
    const int src_id = blockIdx.x * blockDim.x + threadIdx.x;

    _T val = 0;
    if(src_id < _size)
    {
        int connections_count = _graph.get_connections_count(src_id);
        int vector_index = lane_id();
        val = reduce_op(src_id, connections_count, vector_index);
    }

    _T sum = block_reduce_sum(val);
    if (threadIdx.x == 0)
        atomicAdd(_result, sum);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename ReduceOperation, typename GraphContainer>
__global__ void reduce_kernel_sparse(GraphContainer _graph,
                                     const int *_frontier_ids,
                                     const int _frontier_size,
                                     ReduceOperation reduce_op,
                                     _T* _result)
{
    const int frontier_pos = blockIdx.x * blockDim.x + threadIdx.x;

    _T val = 0;
    if(frontier_pos < _frontier_size)
    {
        int src_id = _frontier_ids[frontier_pos];
        int connections_count = _graph.get_connections_count(src_id);
        int vector_index = lane_id();
        val = reduce_op(src_id, connections_count, vector_index);
    }

    _T sum = block_reduce_sum(val);
    if (threadIdx.x == 0)
        atomicAdd(_result, sum);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename ReduceOperation, typename GraphContainer, typename FrontierContainer>
void GraphAbstractionsGPU::reduce_worker_sum(GraphContainer &_graph,
                                             FrontierContainer &_frontier,
                                             ReduceOperation &&reduce_op,
                                             _T &_result)
{
    _T *managed_reduced_result;
    MemoryAPI::allocate_array(&managed_reduced_result, 1);
    managed_reduced_result[0] = 0;
    MemoryAPI::move_array_to_device(managed_reduced_result, 1);
    int vertices_count = _graph.get_vertices_count();

    if(_frontier.get_sparsity_type() == ALL_ACTIVE_FRONTIER)
    {
        SAFE_KERNEL_CALL((reduce_kernel_all_active<<< (vertices_count - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >>>(_graph, vertices_count, reduce_op, managed_reduced_result)));
    }
    else if(_frontier.get_sparsity_type() == DENSE_FRONTIER)
    {
        throw "Error: dense frontier in reduce is not supported";
    }
    else if(_frontier.get_sparsity_type() == SPARSE_FRONTIER)
    {
        int frontier_size = _frontier.get_size();
        SAFE_KERNEL_CALL((reduce_kernel_sparse<<< (frontier_size - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >>>(_graph, _frontier.ids, frontier_size, reduce_op, managed_reduced_result)));
    }

    cudaDeviceSynchronize();
    _T reduce_result = managed_reduced_result[0];

    MemoryAPI::free_array(managed_reduced_result);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


