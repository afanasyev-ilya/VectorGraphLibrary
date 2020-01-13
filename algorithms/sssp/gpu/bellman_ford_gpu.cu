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
#include "../../../architectures.h"
#include <cfloat>
#include <cuda_fp16.h>
#include "../../../graph_representations/base_graph.h"
#include "../../../common_datastructures/gpu_API/gpu_arrays.h"
#include "../../../graph_representations/edges_list_graph/edges_list_graph.h"
#include "../../../graph_representations/vectorised_CSR_graph/vectorised_CSR_graph.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
class VectorisedCSRGraph;

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
                                          const _T *_outgoing_weights,
                                          const int _vertices_count,
                                          _T *_distances,
                                          const int _src_id,
                                          const int _connections_count,
                                          int *_changes)
{
    register const int src_id = _src_id;
    
    register const long long edge_start = _first_part_ptrs[src_id];
    register const long long edge_pos = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(edge_pos < _connections_count)
    {
        register const int dst_id = _outgoing_ids[edge_start + edge_pos];
        register const _T weight = _outgoing_weights[edge_start + edge_pos];
        register _T new_distance = weight + _distances[dst_id];
        
        if(_distances[src_id] > __ldg(&_distances[dst_id]) + weight)
        {
            _distances[src_id] = __ldg(&_distances[dst_id]) + weight;
            _changes[0] = 1;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void __global__ bellman_ford_grid_per_vertex_kernel(const long long *_first_part_ptrs,
                                                    const int *_first_part_sizes,
                                                    const int *_outgoing_ids,
                                                    const _T *_outgoing_weights,
                                                    const int _vertices_count,
                                                    _T *_distances,
                                                    int *_changes,
                                                    const int _vertex_part_start,
                                                    const int _vertex_part_end)
{
    const int src_id = blockIdx.x * blockDim.x + threadIdx.x + _vertex_part_start;
    if(src_id < _vertex_part_end)
    {
        dim3 child_threads(BLOCK_SIZE);
        dim3 child_blocks((_first_part_sizes[src_id] - 1) / BLOCK_SIZE + 1);
        bellman_ford_kernel_child <<< child_blocks, child_threads >>>
             (_first_part_ptrs, _outgoing_ids, _outgoing_weights, _vertices_count, _distances, src_id, _first_part_sizes[src_id], _changes);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void __global__ bellman_ford_block_per_vertex_kernel(const long long *_first_part_ptrs,
                                                     const int *_first_part_sizes,
                                                     const int *_outgoing_ids,
                                                     const _T *_outgoing_weights,
                                                     const int _vertices_count,
                                                     _T *_distances,
                                                     int *_changes,
                                                     const int _vertex_part_start,
                                                     const int _vertex_part_end)
{
    int src_id = blockIdx.x + _vertex_part_start;
    
    if(src_id < _vertex_part_end)
    {
        register const long long edge_start = _first_part_ptrs[src_id];
        register const int connections_count = _first_part_sizes[src_id];
        
        for(register int edge_pos = threadIdx.x; edge_pos < connections_count; edge_pos += BLOCK_SIZE)
        {
            if(edge_pos < connections_count)
            {
                register int dst_id = _outgoing_ids[edge_start + edge_pos];
                register _T weight = _outgoing_weights[edge_start + edge_pos];
                
                if(_distances[src_id] > __ldg(&_distances[dst_id]) + weight)
                {
                    _distances[src_id] = __ldg(&_distances[dst_id]) + weight;
                    _changes[0] = 1;
                }
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void __global__ bellman_ford_warp_per_vertex_kernel(const long long *_first_part_ptrs,
                                                    const int *_first_part_sizes,
                                                    const int *_outgoing_ids,
                                                    const _T *_outgoing_weights,
                                                    const int _vertices_count,
                                                    _T *_distances,
                                                    int *_changes,
                                                    const int _vertex_part_start,
                                                    const int _vertex_part_end)
{
    const register int warp_id = threadIdx.x / WARP_SIZE;
    const register int lane_id = threadIdx.x % WARP_SIZE;
    
    const register int src_id = blockIdx.x * (blockDim.x/ WARP_SIZE) + warp_id + _vertex_part_start;
    
    if(src_id < _vertex_part_end)
    {
        register const long long edge_start = _first_part_ptrs[src_id];
        register const int connections_count = _first_part_sizes[src_id];
        
        for(register int edge_pos = lane_id; edge_pos < connections_count - WARP_SIZE; edge_pos += WARP_SIZE)
        {
            if(edge_pos < connections_count)
            {
                register int dst_id = _outgoing_ids[edge_start + edge_pos];
                register _T weight = _outgoing_weights[edge_start + edge_pos];
                
                if(_distances[src_id] > __ldg(&_distances[dst_id]) + weight)
                {
                    _distances[src_id] = __ldg(&_distances[dst_id]) + weight;
                    _changes[0] = 1;
                }
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void __global__ bellman_ford_remaining_vertices_kernel(const long long *_vector_group_ptrs,
                                                       const int *_vector_group_sizes,
                                                       const int _number_of_vertices_in_first_part,
                                                       const int *_outgoing_ids,
                                                       const _T *_outgoing_weights,
                                                       _T *_distances,
                                                       const int _vertices_count,
                                                       int *_changes)
{
    register const int src_id = blockIdx.x * blockDim.x + threadIdx.x + _number_of_vertices_in_first_part;
    
    if(src_id < _vertices_count)
    {
        register int segment_connections_count  = _vector_group_sizes[blockIdx.x];
        
        if(segment_connections_count > 0)
        {
            register long long segement_edges_start = _vector_group_ptrs[blockIdx.x];
        
            for(register int edge_pos = 0; edge_pos < segment_connections_count; edge_pos++)
            {
                register int dst_id = _outgoing_ids[segement_edges_start + edge_pos * blockDim.x + threadIdx.x];
                register _T weight = _outgoing_weights[segement_edges_start + edge_pos * blockDim.x + threadIdx.x];
                
                if(_distances[src_id] > __ldg(&_distances[dst_id]) + weight)
                {
                    _distances[src_id] = __ldg(&_distances[dst_id]) + weight;
                    _changes[0] = 1;
                }
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void gpu_bellman_ford_wrapper(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                              _TEdgeWeight *_distances,
                              int _source_vertex,
                              int &_iterations_count)
{
    LOAD_VECTORISED_CSR_GRAPH_DATA(_graph);
    
    cudaStream_t grid_processing_stream,block_processing_stream, warp_processing_stream, thread_processing_stream;
    cudaStreamCreate(&grid_processing_stream);
    cudaStreamCreate(&block_processing_stream);
    cudaStreamCreate(&warp_processing_stream);
    cudaStreamCreate(&thread_processing_stream);
    
    init_distances_kernel <<< (vertices_count - 1)/BLOCK_SIZE, BLOCK_SIZE >>> (_distances, vertices_count, _source_vertex);
    
    // device variable to stop iterations, for each source vertex
    int *device_modif;
    int host_modif;
    SAFE_CALL(cudaMalloc((void**)&device_modif, sizeof(int)));
    
    int grid_threshold_start  = 0;
    int grid_threshold_end    = _graph.get_gpu_grid_threshold_vertex();
    int block_threshold_start = grid_threshold_end;
    int block_threshold_end   = _graph.get_gpu_block_threshold_vertex();
    int warp_threshold_start  = block_threshold_end;
    int warp_threshold_end    = _graph.get_gpu_warp_threshold_vertex();
    
    for (int cur_iteration = 0; cur_iteration < vertices_count; cur_iteration++) // do o(|v|) iterations in worst case
    {
        SAFE_CALL(cudaMemset(device_modif, 0, sizeof(int)));
        
        int grid_vertices_count = grid_threshold_end - grid_threshold_start;
        if(grid_vertices_count > 0)
        {
            bellman_ford_grid_per_vertex_kernel <<< grid_vertices_count, 1, 0, grid_processing_stream >>>
            (first_part_ptrs, first_part_sizes, outgoing_ids, outgoing_weights, vertices_count,
             _distances, device_modif, grid_threshold_start, grid_threshold_end);
        }
        
        int block_vertices_count = block_threshold_end - block_threshold_start;
        if(block_vertices_count > 0)
        {
            bellman_ford_block_per_vertex_kernel <<< block_vertices_count, BLOCK_SIZE, 0, block_processing_stream >>>
                (first_part_ptrs, first_part_sizes, outgoing_ids, outgoing_weights, vertices_count,
                 _distances, device_modif, block_threshold_start, block_threshold_end);
        }
        
        int warp_vertices_count = warp_threshold_end - warp_threshold_start;
        if(warp_vertices_count > 0)
        {
            bellman_ford_warp_per_vertex_kernel<<<warp_vertices_count,WARP_SIZE,0,warp_processing_stream>>>
                (first_part_ptrs, first_part_sizes, outgoing_ids, outgoing_weights, vertices_count,
                 _distances, device_modif, warp_threshold_start, warp_threshold_end);
        }
        
        bellman_ford_remaining_vertices_kernel <<< vector_segments_count, VECTOR_LENGTH, 0, thread_processing_stream >>>
                    (vector_group_ptrs, vector_group_sizes, number_of_vertices_in_first_part, outgoing_ids, outgoing_weights,
                     _distances, vertices_count, device_modif);
        cudaDeviceSynchronize();
        
        SAFE_CALL(cudaMemcpy(&host_modif, device_modif, sizeof(int), cudaMemcpyDeviceToHost));
        
        if (host_modif == 0)
        {
            _iterations_count = cur_iteration + 1;
            SAFE_CALL(cudaDeviceSynchronize());
            break;
        }
    }

    SAFE_CALL(cudaFree(device_modif));
    
    cudaStreamDestroy(block_processing_stream);
    cudaStreamDestroy(warp_processing_stream);
    cudaStreamDestroy(thread_processing_stream);
    cudaStreamDestroy(grid_processing_stream);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template void gpu_bellman_ford_wrapper<int, float>(VectorisedCSRGraph<int, float> &_graph, float *_distances, int _source_vertex,
                                                   int &_iterations_count);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* bellman_ford_gpu_cu */
