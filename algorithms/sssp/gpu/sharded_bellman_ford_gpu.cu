//
//  sharded_bellman_ford.cu
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 12/08/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef sharded_bellman_ford_cu
#define sharded_bellman_ford_cu

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include "../../../common_datastructures/gpu_API/cuda_error_handling.h"
#include <cfloat>
#include <cuda_fp16.h>
#include "../../../graph_representations/sharded_graph/shard_CSR/shard_CSR_pointer_data.h"
#include "../../../graph_representations/sharded_graph/shard_vect_CSR/shard_vect_CSR_pointer_data.h"

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define BLOCK_SIZE 1024
#define ELEMENTS_PER_THREAD 8
#define VERTICES_IN_SEGMENT (BLOCK_SIZE*ELEMENTS_PER_THREAD)
#define VECTOR_LENGTH_IN_SHARD 32

enum ShardType {
    SHARD_CSR_TYPE,
    SHARD_VECT_CSR_TYPE
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ init_distances_kernel(float *_distances, int _vertices_count, int _source_vertex);
void __global__ init_distances_kernel(double *_distances, int _vertices_count, int _source_vertex);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void __global__ bellman_ford_sharded_CSR_kernel(ShardCSRPointerData<_T> *_shards_data,
                                                _T *_distances,
                                                int *_changes)
{
    register const int shard_pos = blockIdx.x;
    int src_id = threadIdx.x;
    
    int _vertices_in_shard = _shards_data[shard_pos].vertices_in_shard;
    int *global_src_ids = _shards_data[shard_pos].global_src_ids;
    long long *vertex_ptrs = _shards_data[shard_pos].vertex_ptrs;
    int *dst_ids = _shards_data[shard_pos].dst_ids;
    _T *weights = _shards_data[shard_pos].weights;
    
    __shared__ _T cached_data[VERTICES_IN_SEGMENT];
    
    #pragma unroll(8)
    for(int i = 0; i < ELEMENTS_PER_THREAD; i++)
        cached_data[i*BLOCK_SIZE + threadIdx.x] = _distances[i*BLOCK_SIZE + threadIdx.x + VERTICES_IN_SEGMENT*shard_pos];
    __syncthreads();
    
    while(src_id < _vertices_in_shard)
    {
        _T current_distance = _distances[global_src_ids[src_id]];
        
        for(long long i = vertex_ptrs[src_id]; i < vertex_ptrs[src_id + 1]; i++)
        {
            int dst_id = dst_ids[i] - VERTICES_IN_SEGMENT*shard_pos;
            _T weight = weights[i];
            
            _T dst_distance = cached_data[dst_id] + weight;
            if(current_distance > dst_distance)
            {
                current_distance = dst_distance;
            }
        }
        
        if(current_distance < _distances[global_src_ids[src_id]])
        {
            _distances[global_src_ids[src_id]] = current_distance;
            _changes[0] = 1;
        }
        
        src_id += BLOCK_SIZE;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void __global__ bellman_ford_sharded_vect_CSR_kernel(ShardVectCSRPointerData<_T> *_shards_data,
                                                     _T *_distances,
                                                     int *_changes)
{
    register const int shard_pos = blockIdx.x;
    int src_id = threadIdx.x;
    int i = threadIdx.x % 32;
    
    int vertices_in_shard = _shards_data[shard_pos].vertices_in_shard;
    int *global_src_ids = _shards_data[shard_pos].global_src_ids;
    
    long long *vector_group_ptrs = _shards_data[shard_pos].vector_group_ptrs;
    int *vector_group_sizes = _shards_data[shard_pos].vector_group_sizes;
    
    int *dst_ids = _shards_data[shard_pos].dst_ids;
    _T *weights = _shards_data[shard_pos].weights;
    
    __shared__ _T cached_data[VERTICES_IN_SEGMENT];
    
    #pragma unroll(8)
    for(int i = 0; i < ELEMENTS_PER_THREAD; i++)
        cached_data[i*BLOCK_SIZE + threadIdx.x] = _distances[i*BLOCK_SIZE + threadIdx.x + VERTICES_IN_SEGMENT*shard_pos];
    __syncthreads();
    
    while(src_id < vertices_in_shard)
    {
        int cur_vector_segment = src_id / VECTOR_LENGTH_IN_SHARD;
        long long edge_start = vector_group_ptrs[cur_vector_segment];
        int cur_max_connections_count = vector_group_sizes[cur_vector_segment];
        
        _T current_distance = _distances[global_src_ids[src_id]];
        
        for(int edge_pos = 0; edge_pos < cur_max_connections_count; edge_pos++)
        {
            if(src_id < vertices_in_shard)
            {
                int dst_id = dst_ids[edge_start + edge_pos * VECTOR_LENGTH_IN_SHARD + i] - VERTICES_IN_SEGMENT*shard_pos;
                _T weight = weights[edge_start + edge_pos * VECTOR_LENGTH_IN_SHARD + i];
                _T dst_distance = cached_data[dst_id] + weight;
                
                if(current_distance > dst_distance)
                {
                    current_distance = dst_distance;
                }
            }
        }
        
        if(current_distance < _distances[global_src_ids[src_id]])
        {
            _distances[global_src_ids[src_id]] = current_distance;
            _changes[0] = 1;
        }
        
        src_id += BLOCK_SIZE;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void gpu_sharded_bellman_ford_wrapper(int _number_of_shards,
                                      void *_shards_data,
                                      _T *_distances,
                                      int _vertices_count,
                                      long long _edges_count,
                                      int &_iterations_count,
                                      ShardType _shard_type)
{
    dim3 init_threads(BLOCK_SIZE);
    dim3 init_blocks((_vertices_count - 1) / init_threads.x + 1);
    
    SAFE_KERNEL_CALL(( init_distances_kernel <<< init_blocks, init_threads >>> (_distances, _vertices_count, 0) )); // TODO
    
    dim3 sharded_threads(BLOCK_SIZE);
    dim3 sharded_blocks(_number_of_shards);
    
    int *device_modif;
    int host_modif;
    SAFE_CALL(cudaMalloc((void**)&device_modif, sizeof(int)));
    
    for (int cur_iteration = 0; cur_iteration < _vertices_count; cur_iteration++) // do o(|v|) iterations in worst case
    {
        SAFE_CALL(cudaMemset(device_modif, 0, sizeof(int)));
        
        if(_shard_type == SHARD_CSR_TYPE)
        {
            bellman_ford_sharded_CSR_kernel <<< sharded_blocks, sharded_threads >>> ((ShardCSRPointerData<_T>*)_shards_data, _distances, device_modif);
        }
        else if(_shard_type == SHARD_VECT_CSR_TYPE)
        {
            bellman_ford_sharded_vect_CSR_kernel <<< sharded_blocks, sharded_threads >>> ((ShardVectCSRPointerData<_T>*)_shards_data, _distances, device_modif);
        }
        
        SAFE_CALL(cudaMemcpy(&host_modif, device_modif, sizeof(int), cudaMemcpyDeviceToHost));
        
        _iterations_count++;
        
        if (host_modif == 0)
        {
            SAFE_CALL(cudaDeviceSynchronize());
            break;
        }
    }
    
    SAFE_CALL(cudaFree(device_modif));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template void gpu_sharded_bellman_ford_wrapper<float>(int _number_of_shards,
                                                      void *_shards_data,
                                                      float *_distances,
                                                      int _vertices_count,
                                                      long long _edges_count,
                                                      int &_iterations_count,
                                                      ShardType _shard_type);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* sharded_bellman_ford_cu */
