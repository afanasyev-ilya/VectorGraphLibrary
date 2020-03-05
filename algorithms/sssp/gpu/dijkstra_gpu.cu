#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../../../graph_processing_API/gpu/cuda_API_include.h"

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

    GraphPrimitivesGPU graph_API;

    char *was_updated;
    cudaMalloc((void**)&was_updated, vertices_count*sizeof(char));

    FrontierGPU frontier(vertices_count);

    auto init_op = [_distances, _source_vertex] __device__ (int src_id) {
        if(src_id == _source_vertex)
            _distances[_source_vertex] = 0;
        else
            _distances[src_id] = FLT_MAX;
    };

    auto EMPTY_VERTEX_OP = [] __device__(int src_id, int connections_count){};

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

    graph_API.compute(init_op, vertices_count);
    frontier.filter(_graph, initial_frontier_condition);

    while(frontier.size() > 0)
    {
        cudaMemset(was_updated, 0, sizeof(char) * vertices_count);
        graph_API.advance(_graph, frontier, edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);
        frontier.filter(_graph, frontier_condition);
        _iterations_count++;
    }

    cudaFree(was_updated);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template void gpu_dijkstra_wrapper<int, float>(ExtendedCSRGraph<int, float> &_graph, float *_distances, int _source_vertex,
                                               int &_iterations_count);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
