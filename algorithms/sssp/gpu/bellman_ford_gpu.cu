//
//  bf_gpu.cu
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 01/05/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef bellman_ford_gpu_cu
#define bellman_ford_gpu_cu

#include <iostream>
#include "../../../common_datastructures/gpu_API/cuda_error_handling.h"
#include <cfloat>
#include <cuda_fp16.h>

using namespace std;

#define BLOCK_SIZE 1024
#define NUMBER_OF_CACHED_VERTICES 1024*8

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ init_distances_kernel(float *_distances, int _vertices_count, int _source_vertex)
{
    register const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < _vertices_count)
        _distances[idx] = FLT_MAX;
    
    _distances[_source_vertex] = 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ init_distances_kernel(double *_distances, int _vertices_count, int _source_vertex)
{
    register const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < _vertices_count)
        _distances[idx] = DBL_MAX;
    
    _distances[_source_vertex] = 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void __global__ bellman_ford_kernel_child(const long long *_first_part_ptrs,
                                          const int *_outgoing_ids,
                                          _T *_outgoing_weights,
                                          int _vertices_count,
                                          _T *_distances,
                                          int _src_id,
                                          int _connections_count,
                                          int *_changes)
{
    int src_id = _src_id;
    
    register const long long edge_start = _first_part_ptrs[src_id];
    register const long long edge_pos = blockIdx.x * blockDim.x + threadIdx.x;
    
    register _T reg_distance = _distances[src_id];
    register int reg_changes = 0;
    
    if(edge_pos < _connections_count)
    {
        register const int dst_id = _outgoing_ids[edge_start + edge_pos];
        register const _T weight = _outgoing_weights[edge_start + edge_pos];
        register _T new_distance = weight + _distances[dst_id];
        
        if(reg_distance > new_distance)
        {
            _distances[src_id] = new_distance;
            _changes[0] = 1;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void __global__ bellman_ford_first_vertices_kernel(const long long *_first_part_ptrs,
                                                   int *_first_part_sizes,
                                                   int _number_of_vertices_in_first_part,
                                                   const int *_outgoing_ids,
                                                   _T *_outgoing_weights,
                                                   int _vertices_count,
                                                   _T *_distances,
                                                   int *_changes)
{
    const int src_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(src_id < _number_of_vertices_in_first_part)
    {
        dim3 child_threads(BLOCK_SIZE);
        dim3 child_blocks((_first_part_sizes[src_id] - 1) / BLOCK_SIZE + 1);
        bellman_ford_kernel_child <<< child_blocks, child_threads >>> (_first_part_ptrs, _outgoing_ids, _outgoing_weights,
                                                                   _vertices_count,
                                                                   _distances, src_id, _first_part_sizes[src_id], _changes);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void __global__ bellman_ford_last_vertices_kernel(long long *_vector_group_ptrs,
                                                  int *_vector_group_sizes,
                                                  int _number_of_vertices_in_first_part,
                                                  int *_outgoing_ids,
                                                  _T *_outgoing_weights,
                                                  _T *_distances,
                                                  const _T * __restrict__ _read_only_distances,
                                                  int _vertices_count,
                                                  int *_changes)
{
    register const int src_id = blockIdx.x * blockDim.x + threadIdx.x + _number_of_vertices_in_first_part;
    
    /*__shared__ _T cached_data[NUMBER_OF_CACHED_VERTICES];
    
    for(int i = 0; i < 8; i++)
        cached_data[i * blockDim.x + threadIdx.x] = _distances[i * blockDim.x + threadIdx.x];
    __syncthreads();*/
    
    if(src_id < _vertices_count)
    {
        register int segment_connections_count  = _vector_group_sizes[blockIdx.x];
        
        if(segment_connections_count > 0)
        {
            register long long segement_edges_start = _vector_group_ptrs[blockIdx.x];
            register _T reg_distance = _distances[src_id];
            register int reg_changes = 0;
        
            for(register int edge_pos = 0; edge_pos < segment_connections_count; edge_pos++)
            {
                register int dst_id = _outgoing_ids[segement_edges_start + edge_pos * blockDim.x + threadIdx.x];
                register _T weight = _outgoing_weights[segement_edges_start + edge_pos * blockDim.x + threadIdx.x];
                register _T new_distance = _distances[dst_id] + weight;
                
                if(reg_distance > new_distance)
                {
                    reg_distance = new_distance;
                    _distances[src_id] = reg_distance;
                    reg_changes = 1;
                }
            }
        
            if(reg_changes > 0)
            {
                //_distances[src_id] = reg_distance;
                _changes[0] = 1;
            }
        }
    }
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void gpu_bellman_ford_wrapper(long long *_first_part_ptrs,
                              int *_first_part_sizes,
                              int _vector_segments_count,
                              int _number_of_vertices_in_first_part,
                              long long *_vector_group_ptrs,
                              int *_vector_group_sizes,
                              int *_outgoing_ids,
                              _T *_outgoing_weights,
                              _T *_distances,
                              int _vertices_count,
                              long long _edges_count,
                              int _source_vertex,
                              int &_iterations_count)
{
    dim3 init_threads(BLOCK_SIZE);
    dim3 init_blocks((_vertices_count - 1) / init_threads.x + 1);
    dim3 vertices_threads(256);
    dim3 vertices_blocks(_vector_segments_count);
    
    SAFE_KERNEL_CALL(( init_distances_kernel <<< init_blocks, init_threads >>> (_distances, _vertices_count, _source_vertex) ));
    
    // device variable to stop iterations, for each source vertex
    int *device_modif;
    int host_modif;
    SAFE_CALL(cudaMalloc((void**)&device_modif, sizeof(int)));
    
    for (int cur_iteration = 0; cur_iteration < _vertices_count; cur_iteration++) // do o(|v|) iterations in worst case
    {
        SAFE_CALL(cudaMemset(device_modif, 0, sizeof(int)));
        
        if(_number_of_vertices_in_first_part > 0)
        {
            bellman_ford_first_vertices_kernel
                    <<< (_number_of_vertices_in_first_part - 1)/BLOCK_SIZE + 1, BLOCK_SIZE >>>
                (_first_part_ptrs, _first_part_sizes, _number_of_vertices_in_first_part,
                 _outgoing_ids, _outgoing_weights, _vertices_count, _distances, device_modif);
        }
        
        bellman_ford_last_vertices_kernel <<< vertices_blocks, vertices_threads >>> (_vector_group_ptrs,
                                                                                     _vector_group_sizes,
                                                                                     _number_of_vertices_in_first_part,
                                                                                     _outgoing_ids, _outgoing_weights,
                                                                                     _distances, _distances,
                                                                                     _vertices_count,
                                                                                     device_modif);
        
        SAFE_CALL(cudaMemcpy(&host_modif, device_modif, sizeof(int), cudaMemcpyDeviceToHost));
        
        if (host_modif == 0)
        {
            _iterations_count = cur_iteration + 1;
            SAFE_CALL(cudaDeviceSynchronize());
            break;
        }
    }
    
    SAFE_CALL(cudaFree(device_modif));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template void gpu_bellman_ford_wrapper<float>(long long *_first_part_ptrs, int *_first_part_sizes,
                                              int _vector_segments_count, int _number_of_vertices_in_first_part,
                                              long long *_vector_group_ptrs, int *_vector_group_sizes, int *outgoing_ids,
                                              float *_outgoing_weights, float *device_distances, int _vertices_count,
                                              long long _edges_count, int _source_vertex, int &_iterations_count);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* bellman_ford_gpu_cu */
