#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "frontier_kernels.cu"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FrontierGPU::FrontierGPU(int _vertices_count)
{
    max_frontier_size = _vertices_count;
    current_frontier_size = 0;
    cudaMalloc((void**)&frontier_ids, max_frontier_size*sizeof(int));
    cudaMalloc((void**)&frontier_flags, max_frontier_size*sizeof(char));
    cudaMemset(frontier_ids, 0, max_frontier_size*sizeof(int));
    cudaMemset(frontier_flags, 0, max_frontier_size*sizeof(char));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FrontierGPU::~FrontierGPU()
{
    cudaFree(frontier_ids);
    cudaFree(frontier_flags);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierGPU::set_all_active_frontier()
{
    set_all_active_frontier_kernel<<< max_frontier_size/BLOCK_SIZE, BLOCK_SIZE >>> (frontier_ids, frontier_flags,
            max_frontier_size);
    current_frontier_size = max_frontier_size;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierGPU::split_sorted_frontier(const long long *_vertex_pointers,
                                        int &_grid_threshold_start,
                                        int &_grid_threshold_end,
                                        int &_block_threshold_start,
                                        int &_block_threshold_end,
                                        int &_warp_threshold_start,
                                        int &_warp_threshold_end,
                                        int &_thread_threshold_start,
                                        int &_thread_threshold_end)
{
    int *dev_grid_threshold_vertex;
    int *dev_block_threshold_vertex;
    int *dev_warp_threshold_vertex;
    cudaMalloc((void**)&dev_grid_threshold_vertex, sizeof(int));
    cudaMalloc((void**)&dev_block_threshold_vertex, sizeof(int));
    cudaMalloc((void**)&dev_warp_threshold_vertex, sizeof(int));

    cudaMemset(dev_grid_threshold_vertex, 0, sizeof(int));
    cudaMemset(dev_block_threshold_vertex, 0, sizeof(int));
    cudaMemset(dev_warp_threshold_vertex, 0, sizeof(int));

    split_frontier_kernel<<<(current_frontier_size - 1)/BLOCK_SIZE+1, BLOCK_SIZE>>>(_vertex_pointers, frontier_ids,
            current_frontier_size, dev_grid_threshold_vertex, dev_block_threshold_vertex, dev_warp_threshold_vertex);

    int host_grid_threshold_vertex;
    int host_block_threshold_vertex;
    int host_warp_threshold_vertex;
    cudaMemcpy(&host_grid_threshold_vertex, dev_grid_threshold_vertex, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&host_block_threshold_vertex, dev_block_threshold_vertex, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&host_warp_threshold_vertex, dev_warp_threshold_vertex, sizeof(int), cudaMemcpyDeviceToHost);

    _grid_threshold_start   = 0;
    _grid_threshold_end     = host_grid_threshold_vertex;
    _block_threshold_start  = _grid_threshold_end;
    _block_threshold_end    = host_block_threshold_vertex;
    _warp_threshold_start   = _block_threshold_end;
    _warp_threshold_end     = host_warp_threshold_vertex;
    _thread_threshold_start = _warp_threshold_end;
    _thread_threshold_end   = current_frontier_size;

    cudaFree(dev_grid_threshold_vertex);
    cudaFree(dev_block_threshold_vertex);
    cudaFree(dev_warp_threshold_vertex);
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct is_active
{
    __device__
    bool operator()(const int x)
    {
        return x != -1;
    }
};

struct is_not_active
{
    __device__
    bool operator()(const int x)
    {
        return x == -1;
    }
};

template <typename _TVertexValue, typename _TEdgeWeight, typename Condition>
void FrontierGPU::filter(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                         Condition condition_op)
{
    int vertices_count = _graph.get_vertices_count();

    copy_frontier_ids_kernel<<<(vertices_count-1)/BLOCK_SIZE+1, BLOCK_SIZE>>>(frontier_ids, frontier_flags,
            vertices_count, condition_op);

    int *new_end = thrust::remove_if(thrust::device, frontier_ids, frontier_ids + vertices_count, is_not_active());
    current_frontier_size = new_end - frontier_ids;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////