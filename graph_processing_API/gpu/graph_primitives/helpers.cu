#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void estimate_advance_work_kernel(const int *_frontier_ids,
                                            const int _frontier_size,
                                            const long long *_vertex_pointers,
                                            int* out)
{
    const int frontier_pos = blockIdx.x * blockDim.x + threadIdx.x;

    if(frontier_pos < _frontier_size)
    {
        int src_id = _frontier_ids[frontier_pos];
        int connections_count = _vertex_pointers[src_id + 1] - _vertex_pointers[src_id];

        int sum = connections_count;

        sum = block_reduce_sum(sum);
        if (threadIdx.x == 0)
            atomicAdd(out, sum);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


int GraphPrimitivesGPU::estimate_advance_work(UndirectedCSRGraph &_graph,
                                              FrontierGPU &_frontier)
{
    LOAD_UNDIRECTED_CSR_GRAPH_DATA(_graph);
    long long *vertex_pointers = vertex_pointers;

    int *managed_reduced_result;
    MemoryAPI::allocate_managed_array(&managed_reduced_result, 1);

    int frontier_size = _frontier.size();
    SAFE_KERNEL_CALL((estimate_advance_work_kernel<<< (frontier_size - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >>>(_frontier.ids, frontier_size, vertex_pointers, managed_reduced_result)));

    cudaDeviceSynchronize();
    int reduce_result = managed_reduced_result[0];

    MemoryAPI::free_device_array(managed_reduced_result);

    return reduce_result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
