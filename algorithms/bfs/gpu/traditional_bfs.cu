//
//  traditional_bfs.cu
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 12/06/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef traditional_bfs_cu
#define traditional_bfs_cu

#include <iostream>
#include <vector>
#include "../../../common_datastructures/gpu_API/cuda_error_handling.h"
#include "../../../architectures.h"
#include <cfloat>
#include "../../../graph_representations/base_graph.h"
#include "../../../common_datastructures/gpu_API/gpu_arrays.h"
#include "../../../graph_representations/edges_list_graph/edges_list_graph.h"
#include "../../../graph_representations/extended_CSR_graph/extended_CSR_graph.h"
#include "../change_state.h"

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
GraphStructure gpu_check_graph_structure(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph)
{
    int vertices_count    = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count   ();
    long long    *outgoing_ptrs    = _graph.get_outgoing_ptrs   ();
    
    int portion_of_first_vertices = 0.01 * vertices_count + 1;
    long long number_of_edges_in_first_portion = 0;
    cudaMemcpy(&number_of_edges_in_first_portion, &outgoing_ptrs[portion_of_first_vertices], sizeof(int), cudaMemcpyDeviceToHost);
    
    if((100.0 * number_of_edges_in_first_portion) / edges_count > POWER_LAW_EDGES_THRESHOLD)
        return POWER_LAW_GRAPH;
    else
        return UNIFORM_GRAPH;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__inline__ __device__ int warp_reduce_sum(int val)
{
    for (int i = WARP_SIZE/2; i >= 1; i /= 2)
        val += __shfl_xor_sync(0xffffffff, val, i, WARP_SIZE);
    return val;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__inline__ __device__ int block_reduce_sum(int val)
{
    static __shared__ int shared[32]; // Shared mem for 32 partial sums
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warp_reduce_sum(val);     // Each warp performs partial reduction

    if (lane==0)
        shared[wid]=val; // Write reduced value to shared memory

    __syncthreads();              // Wait for all partial reductions

    //read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    if (wid == 0)
        val = warp_reduce_sum(val); //Final reduce within first warp

    return val;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ init_levels_kernel(int *_device_levels, int _vertices_count, int _source_vertex)
{
    register const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(idx < _vertices_count)
        _device_levels[idx] = UNVISITED_VERTEX;
    
    if(idx == _source_vertex)
        _device_levels[_source_vertex] = 1;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct degree_non_zero
{
    __host__ __device__
    bool operator()(const int x)
    {
        return x > 0;
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ fill_connection_counts_kernel(const long long *_outgoing_ptrs,
                                              int *_sizes_buffer,
                                              int _vertices_count)
{
    register const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < _vertices_count)
    {
        _sizes_buffer[idx] = _outgoing_ptrs[idx + 1] - _outgoing_ptrs[idx];
    }
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int calculate_non_zero_vertices_number(const long long *_outgoing_ptrs,
                                       int *_sizes_buffer,
                                       const int _vertices_count)
{
    fill_connection_counts_kernel<<<(_vertices_count - 1)/BLOCK_SIZE + 1, BLOCK_SIZE>>>(_outgoing_ptrs, _sizes_buffer, _vertices_count);
    
    return count_if(thrust::device, _sizes_buffer, _sizes_buffer + _vertices_count, degree_non_zero());
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ top_down_kernel_child(const long long *_outgoing_ptrs,
                                      const int *_outgoing_ids,
                                      int *_levels,
                                      int _src_id,
                                      const int _connections_count,
                                      const int _current_level,
                                      int *_in_lvl,
                                      int *_vis)
{
    register const long long edge_start = _outgoing_ptrs[_src_id];
    register const long long edge_pos = blockIdx.x * blockDim.x + threadIdx.x;
    
    register int local_vis = 0;
    register int local_in_lvl = _connections_count;
    
    if(edge_pos < _connections_count)
    {
        register int dst_id = _outgoing_ids[edge_start + edge_pos];
        
        int dst_level = __ldg(&_levels[dst_id]);
        if(dst_level == UNVISITED_VERTEX)
        {
            local_vis++;
            _levels[dst_id] = _current_level + 1;
        }
    }
    
    local_vis = block_reduce_sum(local_vis);
    if(threadIdx.x == 0)
    {
        atomicAdd(_vis, local_vis);
    }
    
    if(edge_pos == 0)
    {
        atomicAdd(_in_lvl, local_in_lvl);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ top_down_grid_per_vertex_kernel(const long long *_outgoing_ptrs,
                                                const int *_outgoing_ids,
                                                const int _vertices_count,
                                                int *_levels,
                                                const int *_frontier_ids,
                                                const int _vertex_part_start,
                                                const int _vertex_part_end,
                                                const int _current_level,
                                                int *_in_lvl,
                                                int *_vis)
{
    register const int frontier_pos = blockIdx.x + _vertex_part_start;
    register const int src_id = _frontier_ids[frontier_pos];
    
    if(src_id < _vertex_part_end)
    {
        const int connections_count = _outgoing_ptrs[src_id + 1] - _outgoing_ptrs[src_id];
        
        dim3 child_threads(BLOCK_SIZE);
        dim3 child_blocks((connections_count - 1) / BLOCK_SIZE + 1);
        top_down_kernel_child <<< child_blocks, child_threads >>>
             (_outgoing_ptrs, _outgoing_ids, _levels, src_id, connections_count, _current_level, _in_lvl, _vis);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ top_down_block_per_vertex_kernel(const long long *_outgoing_ptrs,
                                                 const int *_outgoing_ids,
                                                 const int _vertices_count,
                                                 int *_levels,
                                                 const int *_frontier_ids,
                                                 const int _vertex_part_start,
                                                 const int _vertex_part_end,
                                                 const int _current_level,
                                                 int *_in_lvl,
                                                 int *_vis)
{
    register const int frontier_pos = blockIdx.x + _vertex_part_start;
    register int idx = threadIdx.x;
    
    if(frontier_pos < _vertex_part_end)
    {
        register const int src_id = _frontier_ids[frontier_pos];
        register const long long edge_start = _outgoing_ptrs[src_id];
        register const int connections_count = _outgoing_ptrs[src_id + 1] - _outgoing_ptrs[src_id];
        
        register int local_vis = 0;
        register int local_in_lvl = connections_count;
        
        for(register int edge_pos = idx; edge_pos < connections_count; edge_pos += BLOCK_SIZE)
        {
            if(edge_pos < connections_count)
            {
                register int dst_id = _outgoing_ids[edge_start + edge_pos];
                
                int dst_level = __ldg(&_levels[dst_id]);
                if(dst_level == UNVISITED_VERTEX)
                {
                    local_vis++;
                    _levels[dst_id] = _current_level + 1;
                }
            }
        }
        
        local_vis = block_reduce_sum(local_vis);
        
        if(idx == 0)
        {
            atomicAdd(_vis, local_vis);
            atomicAdd(_in_lvl, local_in_lvl);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ top_down_warp_per_vertex_kernel(const long long *_outgoing_ptrs,
                                                const int *_outgoing_ids,
                                                const int _vertices_count,
                                                int *_levels,
                                                const int *_frontier_ids,
                                                const int _vertex_part_start,
                                                const int _vertex_part_end,
                                                const int _current_level,
                                                int *_in_lvl,
                                                int *_vis)
{
    register const int warp_id = threadIdx.x / WARP_SIZE;
    register const int lane_id = threadIdx.x % WARP_SIZE;
    
    register const int frontier_pos = blockIdx.x * (blockDim.x/ WARP_SIZE) + warp_id + _vertex_part_start;
    
    if(frontier_pos < _vertex_part_end)
    {
        register const int src_id = _frontier_ids[frontier_pos];
        register const long long edge_start = _outgoing_ptrs[src_id];
        register const int connections_count = _outgoing_ptrs[src_id + 1] - _outgoing_ptrs[src_id];
        
        register int local_vis = 0;
        register int local_in_lvl = connections_count;
        
        for(register int edge_pos = lane_id; edge_pos < connections_count; edge_pos += WARP_SIZE)
        {
            if(edge_pos < connections_count)
            {
                register int dst_id = _outgoing_ids[edge_start + edge_pos];
                
                int dst_level = __ldg(&_levels[dst_id]);
                if(dst_level == UNVISITED_VERTEX)
                {
                    local_vis++;
                    _levels[dst_id] = _current_level + 1;
                }
            }
        }
        
        local_vis = warp_reduce_sum(local_vis);
        
        if(lane_id == 0)
        {
            atomicAdd(_vis, local_vis);
            atomicAdd(_in_lvl, local_in_lvl);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ top_down_thread_per_vertex_kernel(const long long *_outgoing_ptrs,
                                                  const int *_outgoing_ids,
                                                  const int _vertices_count,
                                                  int *_levels,
                                                  const int *_frontier_ids,
                                                  const int _vertex_part_start,
                                                  const int _vertex_part_end,
                                                  const int _current_level,
                                                  int *_in_lvl,
                                                  int *_vis)
{
    register const int frontier_pos = blockIdx.x * blockDim.x + threadIdx.x + _vertex_part_start;
    
    if(frontier_pos < _vertex_part_end)
    {
        int src_id = _frontier_ids[frontier_pos];
        
        register const long long edge_start = _outgoing_ptrs[src_id];
        register const int connections_count = _outgoing_ptrs[src_id + 1] - _outgoing_ptrs[src_id];
        
        register int local_vis = 0;
        register int local_in_lvl = connections_count;
        
        for(register int edge_pos = 0; edge_pos < connections_count; edge_pos++)
        {
            if(edge_pos < connections_count)
            {
                register int dst_id = _outgoing_ids[edge_start + edge_pos];
                
                int dst_level = __ldg(&_levels[dst_id]);
                if(dst_level == UNVISITED_VERTEX)
                {
                    local_vis++;
                    _levels[dst_id] = _current_level + 1;
                }
            }
        }
        
        atomicAdd(_vis, local_vis);
        atomicAdd(_in_lvl, local_in_lvl);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ bottom_up_vector_extension_kernel(const long long *_outgoing_ptrs,
                                                  const int *_vectorised_outgoing_ids,
                                                  const int _vertices_count,
                                                  int *_levels,
                                                  const int _current_level,
                                                  int *_in_lvl,
                                                  int *_vis)
{
    register const int src_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(src_id < _vertices_count)
    {
        register int local_vis = 0;
        register int local_in_lvl = 0;
            
        if(_levels[src_id] == UNVISITED_VERTEX)
        {
            register const int connections_count = _outgoing_ptrs[src_id + 1] - _outgoing_ptrs[src_id];
            #pragma unroll(4)
            for(register int edge_pos = 0; edge_pos < min(VECTOR_EXTENSION_SIZE,connections_count); edge_pos++)
            {
                register int dst_id = _vectorised_outgoing_ids[src_id + _vertices_count * edge_pos];
                    
                local_in_lvl++;
                int dst_level = __ldg(&_levels[dst_id]);
                if(dst_level == _current_level)
                {
                    _levels[src_id] = _current_level + 1;
                    local_vis++;
                    break;
                }
            }
        }
        
        atomicAdd(_vis, local_vis);
        atomicAdd(_in_lvl, local_in_lvl);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ bottom_up_reminder_warp_per_wertex_kernel(const long long *_outgoing_ptrs,
                                                          const int *_outgoing_ids,
                                                          const int *_frontier_ids,
                                                          const int _frontier_size,
                                                          int *_levels,
                                                          const int _current_level,
                                                          int _vector_extension_edge_start,
                                                          const int _vertex_part_start,
                                                          const int _vertex_part_end,
                                                          int *_in_lvl,
                                                          int *_vis)
{
    register const int warp_id = threadIdx.x / WARP_SIZE;
    register const int lane_id = threadIdx.x % WARP_SIZE;
    
    register const int frontier_pos = blockIdx.x * (blockDim.x/ WARP_SIZE) + warp_id + _vertex_part_start;
    
    if(frontier_pos < _vertex_part_end)
    {
        register const int src_id = _frontier_ids[frontier_pos];
        
        if(_levels[src_id] == UNVISITED_VERTEX)
        {
            register int local_vis = 0;
            register int local_in_lvl = 0;
            
            register const long long edge_start = _outgoing_ptrs[src_id];
            register const int connections_count = _outgoing_ptrs[src_id + 1] - _outgoing_ptrs[src_id];
            
            for(register int edge_pos = _vector_extension_edge_start + lane_id; edge_pos < connections_count; edge_pos += WARP_SIZE)
            {
                if(edge_pos < connections_count)
                {
                    register int dst_id = _outgoing_ids[edge_start + edge_pos];
                    
                    local_in_lvl++;
                    int dst_level = __ldg(&_levels[dst_id]);
                    if(dst_level == _current_level)
                    {
                        _levels[src_id] = _current_level + 1;
                        local_vis++;
                        break;
                    }
                }
            }
            
            atomicAdd(_vis, local_vis);
            atomicAdd(_in_lvl, local_in_lvl);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ bottom_up_reminder_thread_per_wertex_kernel(const long long *_outgoing_ptrs,
                                                            const int *_outgoing_ids,
                                                            const int *_frontier_ids,
                                                            const int _frontier_size,
                                                            int *_levels,
                                                            const int _current_level,
                                                            int _vector_extension_edge_start,
                                                            const int _vertex_part_start,
                                                            const int _vertex_part_end,
                                                            int *_in_lvl,
                                                            int *_vis)
{

    register const int frontier_pos = blockIdx.x * blockDim.x + threadIdx.x + _vertex_part_start;
    
    if(frontier_pos < _vertex_part_end)
    {
        register const int src_id = _frontier_ids[frontier_pos];
        
        if(_levels[src_id] == UNVISITED_VERTEX)
        {
            register int local_vis = 0;
            register int local_in_lvl = 0;
            
            register const long long edge_start = _outgoing_ptrs[src_id];
            register const int connections_count = _outgoing_ptrs[src_id + 1] - _outgoing_ptrs[src_id];
            
            for(register int edge_pos = _vector_extension_edge_start; edge_pos < connections_count; edge_pos++)
            {
                if(edge_pos < connections_count)
                {
                    register int dst_id = _outgoing_ids[edge_start + edge_pos];
                    
                    local_in_lvl++;
                    int dst_level = __ldg(&_levels[dst_id]);
                    if(dst_level == _current_level)
                    {
                        _levels[src_id] = _current_level + 1;
                        local_vis++;
                        break;
                    }
                }
            }
            
            atomicAdd(_vis, local_vis);
            atomicAdd(_in_lvl, local_in_lvl);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct not_zero
{
    __host__ __device__
    bool operator()(const int x)
    {
        return x != UNVISITED_VERTEX;
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ copy_vertex_ids_kernel(const int *_levels,
                                       const int _current_level,
                                       const int _vertices_count,
                                       int *_frontier_ids)
{
    register const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < _vertices_count)
    {
        if(_levels[idx] == _current_level)
            _frontier_ids[idx] = idx;
        else
            _frontier_ids[idx] = UNVISITED_VERTEX;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct level_is_equal
{
    int desired_val;
    level_is_equal(int _desired_val)
    {
        desired_val = _desired_val;
    }
    
    __host__ __device__
    bool operator()(const int x)
    {
        return x == desired_val;
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int calculate_frontier_size(const int *_levels,
                            const int _vertices_count,
                            const int _desired_level)
{
    int frontier_size = count_if(thrust::device, _levels, _levels + _vertices_count, level_is_equal(_desired_level));
    
    return frontier_size;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int generate_next_frontier(const int *_levels,
                           int *_frontier_ids,
                           int *_frontier_vertices_buffer,
                           const int _vertices_count,
                           const int _current_level)
{
    copy_vertex_ids_kernel<<<(_vertices_count-1)/BLOCK_SIZE+1, BLOCK_SIZE>>>(_levels, _current_level, _vertices_count,
                                                                             _frontier_vertices_buffer);
    
    int frontier_size = thrust::count_if(thrust::device, _frontier_vertices_buffer, _frontier_vertices_buffer + _vertices_count, not_zero());

    thrust::copy_if(thrust::device, _frontier_vertices_buffer, _frontier_vertices_buffer + _vertices_count, _frontier_ids, not_zero());
    
    return frontier_size;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ split_frontier_kernel(const long long *_outgoing_ptrs,
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
        
        int current_size = current_size = _outgoing_ptrs[current_id + 1] - _outgoing_ptrs[current_id];;
        int next_size = 0;
        if(idx < (_frontier_size - 1))
        {
            next_size = _outgoing_ptrs[next_id + 1] - _outgoing_ptrs[next_id];
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

void split_frontier(const long long *_outgoing_ptrs,
                    const int *_frontier_ids,
                    const int _frontier_size,
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
    
    split_frontier_kernel<<<(_frontier_size - 1)/BLOCK_SIZE+1, BLOCK_SIZE>>>(_outgoing_ptrs, _frontier_ids,
              _frontier_size, dev_grid_threshold_vertex, dev_block_threshold_vertex, dev_warp_threshold_vertex);
    
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
    _thread_threshold_end   = _frontier_size;
    
    cudaFree(dev_grid_threshold_vertex);
    cudaFree(dev_block_threshold_vertex);
    cudaFree(dev_warp_threshold_vertex);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void top_down_wrapper(const long long *_outgoing_ptrs,
                      const int *_outgoing_ids,
                      const int _vertices_count,
                      int *_levels,
                      int *_frontier_ids,
                      int *_frontier_vertices_buffer,
                      int _frontier_size,
                      const int _current_level,
                      int *_in_lvl,
                      int *_vis,
                      cudaStream_t &_grid_processing_stream,
                      cudaStream_t &_block_processing_stream,
                      cudaStream_t &_warp_processing_stream,
                      cudaStream_t &_thread_processing_stream)
{
    int grid_threshold_start, grid_threshold_end;
    int block_threshold_start, block_threshold_end;
    int warp_threshold_start, warp_threshold_end;
    int thread_threshold_start, thread_threshold_end;
    
    split_frontier(_outgoing_ptrs, _frontier_ids, _frontier_size,
                   grid_threshold_start, grid_threshold_end, block_threshold_start, block_threshold_end,
                   warp_threshold_start, warp_threshold_end, thread_threshold_start, thread_threshold_end);
    
    int grid_vertices_count = grid_threshold_end - grid_threshold_start;
    if(grid_vertices_count > 0)
    {
        top_down_grid_per_vertex_kernel <<< grid_vertices_count, 1, 0, _grid_processing_stream >>>
            (_outgoing_ptrs, _outgoing_ids, _vertices_count, _levels, _frontier_ids, grid_threshold_start,
            grid_threshold_end, _current_level, _in_lvl, _vis);
    }
    
    int block_vertices_count = block_threshold_end - block_threshold_start;
    if(block_vertices_count > 0)
    {
        top_down_block_per_vertex_kernel <<< block_vertices_count, BLOCK_SIZE, 0, _block_processing_stream >>>
             (_outgoing_ptrs, _outgoing_ids, _vertices_count, _levels, _frontier_ids, block_threshold_start,
              block_threshold_end, _current_level, _in_lvl, _vis);
    }
    
    int warp_vertices_count = warp_threshold_end - warp_threshold_start;
    if(warp_vertices_count > 0)
    {
        top_down_warp_per_vertex_kernel <<< WARP_SIZE*(warp_vertices_count - 1)/BLOCK_SIZE + 1, BLOCK_SIZE, 0, _warp_processing_stream >>>
             (_outgoing_ptrs, _outgoing_ids, _vertices_count, _levels, _frontier_ids, warp_threshold_start,
              warp_threshold_end, _current_level, _in_lvl, _vis);
    }
    
    int thread_vertices_count = thread_threshold_end - thread_threshold_start;
    if(thread_vertices_count > 0)
    {
        top_down_thread_per_vertex_kernel <<< (thread_vertices_count - 1)/BLOCK_SIZE + 1, BLOCK_SIZE, 0, _thread_processing_stream >>>
             (_outgoing_ptrs, _outgoing_ids, _vertices_count, _levels, _frontier_ids, thread_threshold_start,
              thread_threshold_end, _current_level, _in_lvl, _vis);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void bottom_up_wrapper(const long long *_outgoing_ptrs,
                       const int *_outgoing_ids,
                       const int *_vectorised_outgoing_ids,
                       const int _vertices_count,
                       const int _non_zero_vertices_count,
                       int *_levels,
                       int *_frontier_ids,
                       int *_frontier_vertices_buffer,
                       int _frontier_size,
                       const int _current_level,
                       int *_in_lvl,
                       int *_vis)
{
    bool use_vect_CSR_extension = false;
    double non_propcessed_vertices = calculate_frontier_size(_levels, _non_zero_vertices_count, -1);
    
    int second_part_start = 0;
    if(non_propcessed_vertices/_non_zero_vertices_count > ENABLE_VECTOR_EXTENSION_THRESHOLD)
    {
        second_part_start = VECTOR_EXTENSION_SIZE;
        bottom_up_vector_extension_kernel<<< (_non_zero_vertices_count - 1)/BLOCK_SIZE + 1, BLOCK_SIZE >>> (_outgoing_ptrs, _vectorised_outgoing_ids, _vertices_count, _levels, _current_level, _in_lvl, _vis);
    }
    
    int reminder_count = generate_next_frontier(_levels, _frontier_ids, _frontier_vertices_buffer, _non_zero_vertices_count, -1);
    
    int grid_threshold_start, grid_threshold_end;
    int block_threshold_start, block_threshold_end;
    int warp_threshold_start, warp_threshold_end;
    int thread_threshold_start, thread_threshold_end;
    split_frontier(_outgoing_ptrs, _frontier_ids,  reminder_count,
                   grid_threshold_start, grid_threshold_end, block_threshold_start, block_threshold_end,
                   warp_threshold_start, warp_threshold_end, thread_threshold_start, thread_threshold_end);
    /*cout << "grid: " << grid_threshold_end - grid_threshold_start << endl;
    cout << "block: " << block_threshold_end - block_threshold_start << endl;
    cout << "warp: " << warp_threshold_end - warp_threshold_start << endl;
    cout << "thread: " << thread_threshold_end - thread_threshold_start << endl;*/
    
    int warp_vertices_count = warp_threshold_end - grid_threshold_start;
    if(warp_vertices_count > 0)
    {
        bottom_up_reminder_warp_per_wertex_kernel<<< WARP_SIZE*(reminder_count - 1)/BLOCK_SIZE + 1, BLOCK_SIZE >>> (_outgoing_ptrs, _outgoing_ids, _frontier_ids, reminder_count, _levels, _current_level, second_part_start, warp_threshold_start, warp_threshold_end, _in_lvl, _vis);
    }
    
    int thread_vertices_count = thread_threshold_end - thread_threshold_start;
    if(thread_vertices_count > 0)
    {
        bottom_up_reminder_thread_per_wertex_kernel<<< (reminder_count - 1)/BLOCK_SIZE + 1, BLOCK_SIZE >>> (_outgoing_ptrs, _outgoing_ids, _frontier_ids, reminder_count, _levels, _current_level, second_part_start, thread_threshold_start, thread_threshold_end, _in_lvl, _vis);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void gpu_direction_optimising_bfs_wrapper(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int *_levels,
                                          int _source_vertex, int &_iterations_count, int *_frontier_ids, int *_frontier_vertices_buffer)
{
    double wall_time = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);
    
    GraphStructure _graph_structure = gpu_check_graph_structure(_graph);
    
    cudaStream_t grid_processing_stream,block_processing_stream, warp_processing_stream, thread_processing_stream;
    cudaStreamCreate(&grid_processing_stream);
    cudaStreamCreate(&block_processing_stream);
    cudaStreamCreate(&warp_processing_stream);
    cudaStreamCreate(&thread_processing_stream);
    
    int current_frontier_size = 1;
    
    int *device_in_lvl, *device_vis;
    cudaMalloc((void**)&device_in_lvl, sizeof(int));
    cudaMalloc((void**)&device_vis, sizeof(int));

    int non_zero_vertices_count = calculate_non_zero_vertices_number(outgoing_ptrs, _frontier_ids, vertices_count);
    
    // init source vertex and initial data
    init_levels_kernel<<<(vertices_count - 1)/BLOCK_SIZE + 1, BLOCK_SIZE>>>(_levels, vertices_count, _source_vertex);
    cudaMemcpy(_frontier_ids, &_source_vertex, sizeof(int), cudaMemcpyHostToDevice);
    
    // run main algorithm
    int current_level = 1;
    int total_visited = 1;
    StateOfBFS current_state = TOP_DOWN;
    while(current_level <= vertices_count)
    {
        StateOfBFS old_state = current_state;
        
        SAFE_CALL(cudaMemset(device_in_lvl, 0, sizeof(int)));
        SAFE_CALL(cudaMemset(device_vis, 0, sizeof(int)));
        
        if(current_state == TOP_DOWN)
        {
            top_down_wrapper(outgoing_ptrs, outgoing_ids, non_zero_vertices_count, _levels, _frontier_ids, _frontier_vertices_buffer,
                             current_frontier_size, current_level, device_in_lvl, device_vis, grid_processing_stream,
                             block_processing_stream, warp_processing_stream, thread_processing_stream);
        }
        else if(current_state == BOTTOM_UP)
        {
            bottom_up_wrapper(outgoing_ptrs, outgoing_ids, vectorised_outgoing_ids, vertices_count, non_zero_vertices_count,
                              _levels, _frontier_ids, _frontier_vertices_buffer, current_frontier_size, current_level, device_in_lvl,
                              device_vis);
        }
        
        int next_frontier_size = calculate_frontier_size(_levels, non_zero_vertices_count, current_level + 1);
        total_visited += next_frontier_size;
        if(next_frontier_size == 0)
            break;

        int in_lvl = 0;
        int vis = 0;
        cudaMemcpy(&in_lvl, device_in_lvl, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&vis, device_vis, sizeof(int), cudaMemcpyDeviceToHost);
        current_state = gpu_change_state(current_frontier_size, next_frontier_size, vertices_count, edges_count, current_state,
                                         vis, in_lvl, current_level, _graph_structure, total_visited);

        if(current_state == TOP_DOWN)
        {
             generate_next_frontier(_levels, _frontier_ids, _frontier_vertices_buffer, non_zero_vertices_count, current_level + 1);
        }
        
        current_level++;
        current_frontier_size = next_frontier_size;
    }
    
    cudaFree(device_in_lvl);
    cudaFree(device_vis);
    
    cudaStreamDestroy(block_processing_stream);
    cudaStreamDestroy(warp_processing_stream);
    cudaStreamDestroy(thread_processing_stream);
    cudaStreamDestroy(grid_processing_stream);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    wall_time += milliseconds / 1000;
    
    //cout << "Time               : " << wall_time << endl;
    //cout << "Performance        : " << ((double)edges_count) / (wall_time * 1e6) << " MFLOPS" << endl << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template void gpu_direction_optimising_bfs_wrapper<int, float>(ExtendedCSRGraph<int, float> &_graph, int *_levels,
                                                               int _source_vertex, int &_iterations_count,
                                                               int *_frontier_ids, int *_frontier_vertices_buffer);
template void gpu_direction_optimising_bfs_wrapper<int, double>(ExtendedCSRGraph<int, double> &_graph, int *_levels,
                                                                int _source_vertex, int &_iterations_count,
                                                                int *_frontier_ids, int *_frontier_vertices_buffer);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* traditional_bfs_cu */
