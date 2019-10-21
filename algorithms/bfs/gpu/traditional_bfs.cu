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
#include <cfloat>
#include "../change_state.h"

#define VECTOR_LENGTH 256
#define BLOCK_SIZE 1024
#define SEGMENTS_IN_BLOCK 4

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ init_distances_kernel(int *_device_levels, int _vertices_count, int _source_vertex)
{
    register const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < _vertices_count)
        _device_levels[idx] = -1;
    
    _device_levels[_source_vertex] = 1;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ top_down_kernel_child(const long long *_first_part_ptrs,
                                      const int *_outgoing_ids,
                                      int _vertices_count,
                                      int *_levels,
                                      int _current_level,
                                      int *_global_in_lvl,
                                      int *_global_vis,
                                      int _src_id,
                                      int _connections_count)
{
    int src_id = _src_id;
    
    register int local_vis = 0;
    
    register const long long edge_start = _first_part_ptrs[src_id];
    register const long long edge_pos = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(edge_pos < _connections_count)
    {
        register const int dst_id = _outgoing_ids[edge_start + edge_pos];
        
        if(_levels[dst_id] == -1)
        {
            local_vis++;
            _levels[dst_id] = _current_level;
        }
        
        if(threadIdx.x == 0)
            atomicAdd(_global_in_lvl, _connections_count);
        atomicAdd(_global_vis, local_vis);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ top_down_kernel(const long long *_first_part_ptrs,
                                int *_first_part_sizes,
                                int _number_of_vertices_in_first_part,
                                int _vector_segments_count,
                                const long long *_vector_group_ptrs,
                                const int *_vector_group_sizes,
                                const int *_outgoing_ids,
                                int _vertices_count,
                                int *_levels,
                                int _current_level,
                                int *_global_in_lvl,
                                int *_global_vis,
                                int *_vertex_queue,
                                int _vertex_queue_size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < _vertex_queue_size)
    {
        int src_id = _vertex_queue[idx];
        if(src_id < _number_of_vertices_in_first_part)
        {
            dim3 child_threads(BLOCK_SIZE);
            dim3 child_blocks((_first_part_sizes[src_id] - 1) / BLOCK_SIZE + 1);
            top_down_kernel_child <<< child_blocks, child_threads >>> (_first_part_ptrs, _outgoing_ids, _vertices_count,
                                                                       _levels, _current_level, _global_in_lvl, _global_vis,
                                                                       src_id, _first_part_sizes[src_id]);
        }
        else
        {
            register int local_vis = 0;
            
            register int cur_vector_segment = (src_id - _number_of_vertices_in_first_part) / VECTOR_LENGTH;
            register int segment_connections_count  = _vector_group_sizes[cur_vector_segment];
            register long long edge_pos = _vector_group_ptrs[cur_vector_segment] + (src_id - _number_of_vertices_in_first_part) % VECTOR_LENGTH;
            
            for(int i = 0; i < segment_connections_count; i++)
            {
                register const int dst_id = _outgoing_ids[edge_pos];
                if(_levels[dst_id] == -1)
                {
                    local_vis++;
                    _levels[dst_id] = _current_level;
                }
                edge_pos += VECTOR_LENGTH;
            }
            
            atomicAdd(_global_vis, local_vis);
            atomicAdd(_global_in_lvl, segment_connections_count);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ bottom_up_kernel_first_vertices(long long *_first_part_ptrs,
                                                int *_first_part_sizes,
                                                int _number_of_vertices_in_first_part,
                                                int *_outgoing_ids,
                                                int _vertices_count,
                                                int *_levels,
                                                int _current_level,
                                                int *_global_in_lvl,
                                                int *_global_vis)
{
    register const int src_id = blockIdx.x;
    register const int idx = threadIdx.x;
    
    register int local_in_lvl = 0;
    register int local_vis = 0;
    
    if(_levels[src_id] == -1)
    {
        local_in_lvl++;
        register const long long edges_start = _first_part_ptrs[src_id];
        register const int connections_count  = _first_part_sizes[src_id];
        
        for(register int edge_pos = idx; edge_pos < connections_count; edge_pos += VECTOR_LENGTH)
        {
            local_in_lvl++;
            int dst_id = _outgoing_ids[edges_start + edge_pos];
            int dst_level = _levels[dst_id];
            
            if(dst_level == (_current_level - 1))
            {
                _levels[src_id] = _current_level;
                local_vis++;
                break;
            }
        }
        
        atomicAdd(_global_vis, local_vis);
        atomicAdd(_global_in_lvl, local_in_lvl);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ bottom_up_kernel_last_vertices(int _vector_segments_count,
                                               const long long *_vector_group_ptrs,
                                               const int *_vector_group_sizes,
                                               const int *_outgoing_ids,
                                               int _number_of_vertices_in_first_part,
                                               int _vertices_count,
                                               int *_levels,
                                               const int * __restrict__ _read_only_levels,
                                               int _current_level,
                                               int *_global_in_lvl,
                                               int *_global_vis)
{
    /*const int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    register const int idx = global_index % VECTOR_LENGTH;
    register const int cur_vector_segment = global_index / VECTOR_LENGTH;
    
    //__shared__ int cached_levels[BLOCK_SIZE];
    //cached_levels[threadIdx.x] = _read_only_levels[threadIdx.x];
    //__syncthreads();
    
    register int local_in_lvl = 0;
    register int local_vis = 0;
    
    register const int segment_first_vertex = cur_vector_segment * VECTOR_LENGTH + _number_of_vertices_in_first_part;
    register const int src_id = segment_first_vertex + idx;
    
    //if((_levels[src_id] == -1) && (src_id < _vertices_count))
    if(src_id < _vertices_count)
    {
        long long segement_edges_start = _vector_group_ptrs[cur_vector_segment];
        int segment_connections_count  = _vector_group_sizes[cur_vector_segment];
        
        for(int edge_pos = 0; edge_pos < segment_connections_count; edge_pos++)
        {
            local_in_lvl++;
            int dst_id = _outgoing_ids[segement_edges_start + edge_pos * VECTOR_LENGTH + idx];
            //int dst_level = 0;
            //if(dst_id < BLOCK_SIZE)
            //    dst_level = cached_levels[dst_id];
            //else
            //    dst_level = __ldg(&_read_only_levels[dst_id]);
            
            if((_levels[dst_id] == (_current_level - 1)) && (_levels[src_id] == -1))
            {
                _levels[src_id] = _current_level;
                local_vis++;
                //break;
            }
        }
        atomicAdd(_global_vis, local_vis);
        atomicAdd(_global_in_lvl, local_in_lvl);
    }*/
    
    register const int idx = threadIdx.x;
    register const int cur_vector_segment = blockIdx.x;
    
    register int local_in_lvl = 0;
    register int local_vis = 0;
    
    register const int segment_first_vertex = cur_vector_segment * blockDim.x + _number_of_vertices_in_first_part;
    register const int src_id = segment_first_vertex + idx;
    
    register const int src_level = _levels[src_id];
    if(src_level == -1)
    {
        int segment_connections_count  = _vector_group_sizes[cur_vector_segment];
        
        if(segment_connections_count > 0)
        {
            long long segement_edges_start = _vector_group_ptrs[cur_vector_segment];
            long long edge_pos = segement_edges_start + idx;
            
            #pragma unroll 32
            for(int i = 0; i < segment_connections_count; i++)
            {
                local_in_lvl++;
                int dst_id = _outgoing_ids[edge_pos];
                int dst_level = _levels[dst_id];
                
                if(dst_level == (_current_level - 1))
                {
                    _levels[src_id] = _current_level;
                    local_vis++;
                    break;
                }
                
                edge_pos += VECTOR_LENGTH;
            }
            atomicAdd(_global_vis, local_vis);
            atomicAdd(_global_in_lvl, local_in_lvl);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>

template <typename T>
struct is_active : public thrust::unary_function<T,bool>
{
    __host__ __device__
    bool operator()(T x)
    {
        return x > 0;
    }
};

void __global__ copy_active_vertices_ids(int *_status_array,
                                         int _current_level,
                                         int _vertices_count,
                                         int *_output_array)
{
    register const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < _vertices_count)
    {
        if(_status_array[idx] == _current_level)
            _output_array[idx] = idx;
        else
            _output_array[idx] = -1;
    }
}

int convert_status_array_to_vertex_queue(int *_status_array, int _current_level, int _vertices_count, int *_vertex_queue,
                                         int *_tmp_array)
{
    dim3 threads(BLOCK_SIZE);
    dim3 blocks(_vertices_count / BLOCK_SIZE);
    
    copy_active_vertices_ids <<< blocks, threads >>> (_status_array, _current_level, _vertices_count, _tmp_array);
    
    thrust::device_ptr<int> thrust_tmp_array = thrust::device_pointer_cast(_tmp_array);
    thrust::device_ptr<int> thrust_vertex_queue = thrust::device_pointer_cast(_vertex_queue);
    
    thrust::device_ptr<int> indices_end = thrust::copy_if(thrust::device, _tmp_array, _tmp_array + _vertices_count, thrust_vertex_queue, is_active<int>());
    
    int frontier_size = indices_end - thrust_vertex_queue;
    return frontier_size;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void gpu_direction_optimising_bfs_wrapper(long long *_first_part_ptrs,
                                          int *_first_part_sizes,
                                          int _vector_segments_count,
                                          long long *_vector_group_ptrs,
                                          int *_vector_group_sizes,
                                          int *_outgoing_ids,
                                          int _number_of_vertices_in_first_part,
                                          int *_levels,
                                          int _vertices_count,
                                          long long _edges_count,
                                          int _source_vertex)
{
    dim3 init_threads(VECTOR_LENGTH);
    dim3 init_blocks((_vertices_count - 1) / init_threads.x + 1);
    
    SAFE_KERNEL_CALL(( init_distances_kernel <<< init_blocks, init_threads >>> (_levels, _vertices_count, _source_vertex) ));
    
    int *device_in_lvl, *device_vis;
    SAFE_CALL(cudaMalloc((void**)&device_in_lvl, sizeof(int)));
    SAFE_CALL(cudaMalloc((void**)&device_vis, sizeof(int)));
    
    vector<int> level_num;
    vector<double> level_perf;
    vector<string> level_state;
    vector<long long> level_edges_checked;
    
    int *vertex_queue;
    int *tmp_array;
    SAFE_CALL(cudaMalloc((void**)&vertex_queue, _vertices_count*sizeof(int)));
    SAFE_CALL(cudaMalloc((void**)&tmp_array, _vertices_count*sizeof(int)));
    
    int vertex_queue_size = convert_status_array_to_vertex_queue(_levels, 1, _vertices_count, vertex_queue, tmp_array);
    
    StateOfBFS current_state = TOP_DOWN;
    double total_time = 0;
    int current_level = 2;
    int old_vertex_queue_size = 1;
    int old_vis = 1;
    while(true)
    {
        StateOfBFS old_state = current_state;
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        
        //int current_queue_size = global_queue.get_size();
        SAFE_CALL(cudaMemset(device_in_lvl, 0, sizeof(int)));
        SAFE_CALL(cudaMemset(device_vis, 0, sizeof(int)));
        
        if(current_state == TOP_DOWN)
        {
            SAFE_KERNEL_CALL(( top_down_kernel <<< (vertex_queue_size - 1)/BLOCK_SIZE + 1, BLOCK_SIZE >>> (_first_part_ptrs,
                             _first_part_sizes, _number_of_vertices_in_first_part,
                             _vector_segments_count, _vector_group_ptrs, _vector_group_sizes, _outgoing_ids, _vertices_count,
                             _levels, current_level, device_in_lvl, device_vis, vertex_queue, vertex_queue_size) ));
        }
        else
        {
            if(_number_of_vertices_in_first_part > 0)
            {
                bottom_up_kernel_first_vertices <<< _number_of_vertices_in_first_part, VECTOR_LENGTH >>> (_first_part_ptrs, _first_part_sizes, _number_of_vertices_in_first_part, _outgoing_ids, _vertices_count, _levels, current_level, device_in_lvl, device_vis);
            }
        
            //dim3 bu_threads(BLOCK_SIZE);
            //dim3 bu_blocks(_vertices_count / BLOCK_SIZE);
            
            dim3 bu_threads(VECTOR_LENGTH);
            dim3 bu_blocks(_vector_segments_count);
            
            SAFE_KERNEL_CALL(( bottom_up_kernel_last_vertices<<< bu_blocks, bu_threads >>> (_vector_segments_count,
                                                                         _vector_group_ptrs, _vector_group_sizes,
                                                                         _outgoing_ids, _number_of_vertices_in_first_part,
                                                                         _vertices_count, _levels, _levels, current_level,
                                                                         device_in_lvl, device_vis) ));
        }
        
        int in_lvl = 0, vis = 0;
        SAFE_CALL(cudaMemcpy(&in_lvl, device_in_lvl, sizeof(int), cudaMemcpyDeviceToHost));
        SAFE_CALL(cudaMemcpy(&vis, device_vis, sizeof(int), cudaMemcpyDeviceToHost));
        current_state = gpu_change_state(old_vis, vis, _vertices_count, _edges_count, current_state, vis, in_lvl,
                                         current_level);
        
        if((current_state == TOP_DOWN) && (vis != 0))
        {
            vertex_queue_size = convert_status_array_to_vertex_queue(_levels, current_level, _vertices_count, vertex_queue,
                                                                     tmp_array);
        }
        old_vis = vis;
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_time += milliseconds;
        
        cout << "cur level: " << current_level << endl;
        //cout << "number of active vertices: " << active_vertices_num << ", " << 100.0*(double)active_vertices_num/((double)_vertices_count) << " %" << endl;
        cout << "in_lvl: "<< in_lvl << " vs " << _edges_count << endl;
        cout << "time: " << milliseconds << " ms" << endl;
        cout << "global bandwidth: " << (2.0 * sizeof(int) * _edges_count) / (milliseconds * 1e6) << " GB/s" << endl;
        cout << "real bandwidth: " << (2.0 * sizeof(int) * in_lvl) / (milliseconds * 1e6) << " GB/s" << endl << endl;
        
        current_level++;
        
        level_num.push_back(current_level - 1);
        level_perf.push_back(milliseconds);
        level_edges_checked.push_back(in_lvl);
        if(old_state == TOP_DOWN)
            level_state.push_back("TD");
        else
            level_state.push_back("BU");
        
        if(vis == 0)
            break;
    }
    
    double edges_sum = 0.0;
    cout << "GPU BFS perf: " << ((double)_edges_count) / (total_time * 1e3) << " MFLOPS" " ms" << endl;
    for(int i = 0; i < level_perf.size(); i++)
    {
        cout << "level " << level_num[i] << " in " << level_state[i] <<  " | perf: " << ((double)_edges_count) / (level_perf[i] * 1e6) << " GFLOPS " << level_perf[i] << " ms  | " << level_edges_checked[i] << endl;
        edges_sum += level_edges_checked[i];
    }
    cout << "total edges: " << _edges_count << endl;
    cout << "BFS checked: " << (edges_sum / ((double)_edges_count)) * 100.0 << " % of graph edges" << endl;
    
    SAFE_CALL(cudaFree(device_in_lvl));
    SAFE_CALL(cudaFree(device_vis));
    SAFE_CALL(cudaFree(vertex_queue));
    SAFE_CALL(cudaFree(tmp_array));
}

void gpu_scan_test()
{
    int size = 100000;
    thrust::device_vector<float> vector(size);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    thrust::inclusive_scan(vector.begin(), vector.end(), vector.begin());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float time = milliseconds / 1000.0;
    cout << "gpu band: " << (((double)size) * ((double)sizeof(float)) * 2.0) / ((milliseconds) * 1e9) << " GB/s" << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* traditional_bfs_cu */
