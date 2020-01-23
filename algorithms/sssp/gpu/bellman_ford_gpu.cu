//
//  bf_gpu.cu
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 01/05/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef bellman_ford_gpu_cu
#define bellman_ford_gpu_cu

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include "../../../common_datastructures/gpu_API/cuda_error_handling.h"
#include "../../../architectures.h"
#include <cfloat>
#include <cuda_fp16.h>
#include "../../../graph_representations/base_graph.h"
#include "../../../common_datastructures/gpu_API/gpu_arrays.h"
#include "../../../graph_representations/edges_list_graph/edges_list_graph.h"
#include "../../../graph_representations/vectorised_CSR_graph/vectorised_CSR_graph.h"
#include "../../../graph_processing_API/gpu/graph_primitives_gpu.cuh"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ init_distances_kernel(float *_distances, int _vertices_count, int _source_vertex)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(idx < _vertices_count)
        _distances[idx] = FLT_MAX;

    if(idx == _source_vertex)
        _distances[_source_vertex] = 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void gpu_bellman_ford_wrapper(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                              _TEdgeWeight *_distances,
                              int _source_vertex,
                              int &_iterations_count)
{
    LOAD_VECTORISED_CSR_GRAPH_DATA(_graph);

    GraphPrimitivesGPU operations;

    int *changes;
    cudaMallocManaged(&changes, sizeof(int));

    auto init_op = [_distances, _source_vertex] __device__ (int src_id) {
        if(src_id == _source_vertex)
            _distances[_source_vertex] = 0;
        else
            _distances[src_id] = FLT_MAX;
    };

    auto vertex_preprocess_op = [] __device__(int src_id, int connections_count){};
    auto vertex_postprocess_op = [] __device__(int src_id, int connections_count){};

    auto edge_op = [outgoing_weights, _distances, changes] __device__(int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos, int connections_count){
        _TEdgeWeight weight = outgoing_weights[global_edge_pos];
        _TEdgeWeight dst_weight = __ldg(&_distances[dst_id]);
        _TEdgeWeight src_weight = __ldg(&_distances[src_id]);

        if(dst_weight > src_weight + weight)
        {
            _distances[dst_id] = src_weight + weight;
            changes[0] = 1;
        }
    };

    operations.init(_graph.get_vertices_count(), init_op);
    
    for (int cur_iteration = 0; cur_iteration < vertices_count; cur_iteration++) // do o(|v|) iterations in worst case
    {
        changes[0] = 0;

        operations.advance(_graph, edge_op, vertex_preprocess_op, vertex_postprocess_op);

        if (changes[0] == 0)
        {
            _iterations_count = cur_iteration + 1;
            break;
        }
    }

    cudaFree(changes);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template void gpu_bellman_ford_wrapper<int, float>(VectorisedCSRGraph<int, float> &_graph, float *_distances, int _source_vertex,
                                                   int &_iterations_count);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* bellman_ford_gpu_cu */
