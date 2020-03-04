#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <ctime>
#include "../../../common/gpu_API/cuda_error_handling.h"
#include "../../../architectures.h"
#include <cfloat>
#include <cuda_fp16.h>
#include "../../../graph_representations/base_graph.h"
#include "../../../graph_representations/edges_list_graph/edges_list_graph.h"
#include "../../../graph_representations/extended_CSR_graph/extended_CSR_graph.h"
#include "../../../graph_processing_API/gpu/graph_primitives/graph_primitives_gpu.cuh"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void gpu_dijkstra_wrapper(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                          _TEdgeWeight *_distances,
                          int _source_vertex,
                          int &_iterations_count)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    GraphPrimitivesGPU operations;

    char *was_updated;
    cudaMalloc((void**)&was_updated, vertices_count*sizeof(char));

    FrontierGPU frontier(_graph.get_vertices_count());

    auto init_op = [_distances, _source_vertex] __device__ (int src_id) {
        if(src_id == _source_vertex)
            _distances[_source_vertex] = 0;
        else
            _distances[src_id] = FLT_MAX;
    };

    auto vertex_preprocess_op = [] __device__(int src_id, int connections_count){};
    auto vertex_postprocess_op = [] __device__(int src_id, int connections_count){};

    auto edge_op = [outgoing_weights, _distances, was_updated] __device__(int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos, int connections_count){
        _TEdgeWeight weight = outgoing_weights[global_edge_pos];
        _TEdgeWeight src_weight = __ldg(&_distances[src_id]);
        _TEdgeWeight dst_weight = __ldg(&_distances[dst_id]);

        if(dst_weight > src_weight + weight)
        {
            _distances[dst_id] = src_weight + weight;
            was_updated[dst_id] = 1;
            was_updated[src_id] = 1;
        }
    };

    auto initial_frontier_condition = [_source_vertex] __device__(int idx)->bool{
        if(idx == _source_vertex)
            return true;
        else
            return false;
    };

    auto frontier_condition = [was_updated] __device__(int idx)->bool{
        if(was_updated[idx] > 0)
            return true;
        else
            return false;
    };

    operations.init(_graph.get_vertices_count(), init_op);
    frontier.generate_frontier(_graph, initial_frontier_condition);

    while(frontier.size() > 0)
    {
        cudaMemset(was_updated, 0, sizeof(char) * vertices_count);
        operations.advance(_graph, frontier, edge_op, vertex_preprocess_op, vertex_postprocess_op);
        frontier.generate_frontier(_graph, frontier_condition);
        _iterations_count++;
    }

    cudaFree(was_updated);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template void gpu_dijkstra_wrapper<int, float>(ExtendedCSRGraph<int, float> &_graph, float *_distances, int _source_vertex,
                                               int &_iterations_count);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
