#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ set_all_active_frontier_kernel(int *_frontier_ids, char *_frontier_flags, int _vertices_count)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < _vertices_count)
    {
        _frontier_ids[idx] = idx;
        _frontier_flags[idx] = GPU_IN_FRONTIER_FLAG;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ split_frontier_kernel(const long long *_vertex_pointers,
                                      const int *_frontier_ids,
                                      const int _frontier_size,
                                      int *_grid_threshold_vertex,
                                      int *_block_threshold_vertex,
                                      int *_warp_threshold_vertex)
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

        if((current_size > GPU_GRID_THREASHOLD_VALUE) && (next_size <= GPU_GRID_THREASHOLD_VALUE))
        {
            *_grid_threshold_vertex = idx + 1;
        }
        if((current_size > GPU_BLOCK_THREASHOLD_VALUE) && (next_size <= GPU_BLOCK_THREASHOLD_VALUE))
        {
            *_block_threshold_vertex = idx + 1;
        }
        if((current_size > GPU_WARP_THREASHOLD_VALUE) && (next_size <= GPU_WARP_THREASHOLD_VALUE))
        {
            *_warp_threshold_vertex = idx + 1;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Condition>
void __global__ copy_frontier_ids_kernel(int *_frontier_ids,
                                         char *_frontier_flags,
                                         const int _vertices_count,
                                         Condition condition_op)
{
    register const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < _vertices_count)
    {
        if(condition_op(idx) == true)
        {
            _frontier_ids[idx] = idx;
            _frontier_flags[idx] = GPU_IN_FRONTIER_FLAG;
        }
        else
        {
            _frontier_ids[idx] = -1;
            _frontier_flags[idx] = GPU_NOT_IN_FRONTIER_FLAG;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////