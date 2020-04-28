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

    SAFE_KERNEL_CALL((set_all_active_frontier_kernel<<< (max_size - 1)/BLOCK_SIZE + 1, BLOCK_SIZE >>> (ids, flags, max_size)));
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
                                        int &_vwp_16_threshold_start,
                                        int &_vwp_16_threshold_end,
                                        int &_vwp_8_threshold_start,
                                        int &_vwp_8_threshold_end,
                                        int &_vwp_4_threshold_start,
                                        int &_vwp_4_threshold_end,
                                        int &_vwp_2_threshold_start,
                                        int &_vwp_2_threshold_end,
                                        int &_thread_threshold_start,
                                        int &_thread_threshold_end)
{
    int *grid_threshold_vertex;
    int *block_threshold_vertex;
    int *warp_threshold_vertex;
    int *vwp_16_threshold_vertex;
    int *vwp_8_threshold_vertex;
    int *vwp_4_threshold_vertex;
    int *vwp_2_threshold_vertex;

    MemoryAPI::allocate_managed_array(&grid_threshold_vertex, 1);
    MemoryAPI::allocate_managed_array(&block_threshold_vertex, 1);
    MemoryAPI::allocate_managed_array(&warp_threshold_vertex, 1);
    MemoryAPI::allocate_managed_array(&vwp_16_threshold_vertex, 1);
    MemoryAPI::allocate_managed_array(&vwp_8_threshold_vertex, 1);
    MemoryAPI::allocate_managed_array(&vwp_4_threshold_vertex, 1);
    MemoryAPI::allocate_managed_array(&vwp_2_threshold_vertex, 1);

    grid_threshold_vertex[0] = 0;
    block_threshold_vertex[0] = 0;
    warp_threshold_vertex[0] = 0;
    vwp_16_threshold_vertex[0] = 0;
    vwp_8_threshold_vertex[0] = 0;
    vwp_4_threshold_vertex[0] = 0;
    vwp_2_threshold_vertex[0] = 0;

    split_frontier_kernel<<<(current_size - 1)/BLOCK_SIZE+1, BLOCK_SIZE>>>(_vertex_pointers, ids,
            current_size, grid_threshold_vertex, block_threshold_vertex, warp_threshold_vertex,
            vwp_16_threshold_vertex, vwp_8_threshold_vertex, vwp_4_threshold_vertex, vwp_2_threshold_vertex);

    cudaDeviceSynchronize();

    _grid_threshold_start   = 0;
    _grid_threshold_end     = grid_threshold_vertex[0];
    _block_threshold_start  = _grid_threshold_end;
    _block_threshold_end    = block_threshold_vertex[0];
    _warp_threshold_start   = _block_threshold_end;
    _warp_threshold_end     = warp_threshold_vertex[0];
    _vwp_16_threshold_start = _warp_threshold_end;
    _vwp_16_threshold_end   = vwp_16_threshold_vertex[0];
    _vwp_8_threshold_start = _vwp_16_threshold_end;
    _vwp_8_threshold_end   = vwp_8_threshold_vertex[0];
    _vwp_4_threshold_start = _vwp_8_threshold_end;
    _vwp_4_threshold_end   = vwp_4_threshold_vertex[0];
    _vwp_2_threshold_start = _vwp_4_threshold_end;
    _vwp_2_threshold_end   = vwp_2_threshold_vertex[0];
    _thread_threshold_start = _vwp_2_threshold_end;
    _thread_threshold_end   = current_size;

    MemoryAPI::free_device_array(grid_threshold_vertex);
    MemoryAPI::free_device_array(block_threshold_vertex);
    MemoryAPI::free_device_array(warp_threshold_vertex);
    MemoryAPI::free_device_array(vwp_16_threshold_vertex);
    MemoryAPI::free_device_array(vwp_8_threshold_vertex);
    MemoryAPI::free_device_array(vwp_4_threshold_vertex);
    MemoryAPI::free_device_array(vwp_2_threshold_vertex);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////