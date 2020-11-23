#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FrontierGPU::FrontierGPU(VectCSRGraph &_graph, TraversalDirection _direction)
{
    max_size = _graph.get_vertices_count();
    direction = _direction;
    graph_ptr = &_graph;
    init();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FrontierGPU::~FrontierGPU()
{
    cudaFree(ids);
    cudaFree(flags);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierGPU::init()
{
    MemoryAPI::allocate_array(&ids, max_size);
    MemoryAPI::allocate_array(&flags, max_size);
    cudaMemset(ids, 0, max_size*sizeof(int));
    cudaMemset(flags, 0, max_size*sizeof(int));
    type = ALL_ACTIVE_FRONTIER;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ split_frontier_kernel(const long long *_vertex_pointers,
                                      const int *_frontier_ids,
                                      const int _frontier_size,
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

void FrontierGPU::split_sorted_frontier(const long long *_vertex_pointers,
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

    MemoryAPI::allocate_array(&grid_threshold_vertex, 1);
    MemoryAPI::allocate_array(&block_threshold_vertex, 1);
    MemoryAPI::allocate_array(&warp_threshold_vertex, 1);
    MemoryAPI::allocate_array(&vwp_16_threshold_vertex, 1);
    MemoryAPI::allocate_array(&vwp_8_threshold_vertex, 1);
    MemoryAPI::allocate_array(&vwp_4_threshold_vertex, 1);
    MemoryAPI::allocate_array(&vwp_2_threshold_vertex, 1);

    block_threshold_vertex[0] = 0;
    warp_threshold_vertex[0] = 0;
    vwp_16_threshold_vertex[0] = 0;
    vwp_8_threshold_vertex[0] = 0;
    vwp_4_threshold_vertex[0] = 0;
    vwp_2_threshold_vertex[0] = 0;

    split_frontier_kernel<<<(current_size - 1)/BLOCK_SIZE+1, BLOCK_SIZE>>>(_vertex_pointers, ids,
            current_size, block_threshold_vertex, warp_threshold_vertex,
            vwp_16_threshold_vertex, vwp_8_threshold_vertex, vwp_4_threshold_vertex, vwp_2_threshold_vertex);

    cudaDeviceSynchronize();

    _block_threshold_start  = 0;
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

    MemoryAPI::free_array(grid_threshold_vertex);
    MemoryAPI::free_array(block_threshold_vertex);
    MemoryAPI::free_array(warp_threshold_vertex);
    MemoryAPI::free_array(vwp_16_threshold_vertex);
    MemoryAPI::free_array(vwp_8_threshold_vertex);
    MemoryAPI::free_array(vwp_4_threshold_vertex);
    MemoryAPI::free_array(vwp_2_threshold_vertex);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
