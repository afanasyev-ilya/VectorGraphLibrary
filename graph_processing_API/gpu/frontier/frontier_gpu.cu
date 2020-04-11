#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "frontier_kernels.cu"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FrontierGPU::FrontierGPU(int _vertices_count)
{
    max_size = _vertices_count;
    current_size = 0;
    cudaMalloc((void**)&ids, max_size*sizeof(int));
    cudaMalloc((void**)&flags, max_size*sizeof(int));
    cudaMemset(ids, 0, max_size*sizeof(int));
    cudaMemset(flags, 0, max_size*sizeof(int));
    type = ALL_ACTIVE_FRONTIER;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FrontierGPU::~FrontierGPU()
{
    cudaFree(ids);
    cudaFree(flags);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierGPU::set_all_active()
{
    type = ALL_ACTIVE_FRONTIER;

    SAFE_KERNEL_CALL((set_all_active_frontier_kernel<<< max_size/BLOCK_SIZE, BLOCK_SIZE >>> (ids, flags, max_size)));
    current_size = max_size;
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

    split_frontier_kernel<<<(current_size - 1)/BLOCK_SIZE+1, BLOCK_SIZE>>>(_vertex_pointers, ids,
            current_size, dev_grid_threshold_vertex, dev_block_threshold_vertex, dev_warp_threshold_vertex);

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
    _thread_threshold_end   = current_size;

    cudaFree(dev_grid_threshold_vertex);
    cudaFree(dev_block_threshold_vertex);
    cudaFree(dev_warp_threshold_vertex);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////