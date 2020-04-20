#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../../../architectures.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ set_all_active_frontier_kernel(int *_frontier_ids, int *_frontier_flags, int _vertices_count)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < _vertices_count)
    {
        _frontier_ids[idx] = idx;
        _frontier_flags[idx] = IN_FRONTIER_FLAG;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ split_frontier_kernel(const long long *_vertex_pointers,
                                      const int *_frontier_ids,
                                      const int _frontier_size,
                                      int *_grid_threshold_vertex,
                                      int *_block_threshold_vertex,
                                      int *_warp_threshold_vertex,
                                      int *_vwp_16_threshold_vertex,
                                      int *_vwp_8_threshold_vertex,
                                      int *_vwp_4_threshold_vertex,
                                      int *_vwp_2_threshold_vertex)
{
    register const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < _frontier_size)
    {
        const int current_id = _frontier_ids[idx];
        const int next_id = _frontier_ids[idx+1];

        int current_size = _vertex_pointers[current_id + 1] - _vertex_pointers[current_id];;
        int next_size = 0;
        if(idx < (_frontier_size - 1))
        {
            next_size = _vertex_pointers[next_id + 1] - _vertex_pointers[next_id];
        }

        if((current_size > GPU_GRID_THRESHOLD_VALUE) && (next_size <= GPU_GRID_THRESHOLD_VALUE))
        {
            *_grid_threshold_vertex = idx + 1;
        }
        if((current_size > GPU_BLOCK_THRESHOLD_VALUE) && (next_size <= GPU_BLOCK_THRESHOLD_VALUE))
        {
            *_block_threshold_vertex = idx + 1;
        }
        if((current_size > GPU_WARP_THRESHOLD_VALUE) && (next_size <= GPU_WARP_THRESHOLD_VALUE))
        {
            *_warp_threshold_vertex = idx + 1;
        }

        if((current_size > GPU_VWP_16_THRESHOLD_VALUE) && (next_size <= GPU_VWP_16_THRESHOLD_VALUE))
        {
            *_vwp_16_threshold_vertex = idx + 1;
        }
        if((current_size > GPU_VWP_8_THRESHOLD_VALUE) && (next_size <= GPU_VWP_8_THRESHOLD_VALUE))
        {
            *_vwp_8_threshold_vertex = idx + 1;
        }
        if((current_size > GPU_VWP_4_THRESHOLD_VALUE) && (next_size <= GPU_VWP_4_THRESHOLD_VALUE))
        {
            *_vwp_4_threshold_vertex = idx + 1;
        }
        if((current_size > GPU_VWP_2_THRESHOLD_VALUE) && (next_size <= GPU_VWP_2_THRESHOLD_VALUE))
        {
            *_vwp_2_threshold_vertex = idx + 1;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////