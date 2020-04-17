#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../../../graph_processing_API/gpu/cuda_API_include.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void gpu_dijkstra_all_active_wrapper(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                     _TEdgeWeight *_distances,
                                     int _source_vertex,
                                     int &_iterations_count,
                                     TraversalDirection _traversal_direction)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);
    GraphPrimitivesGPU graph_API;
    FrontierGPU frontier(_graph.get_vertices_count());

    frontier.set_all_active();

    auto init_op = [_distances, _source_vertex] __device__ (int src_id, int connections_count) {
        if(src_id == _source_vertex)
            _distances[_source_vertex] = 0;
        else
            _distances[src_id] = FLT_MAX;
    };

    graph_API.compute(_graph, frontier, init_op);

    int *changes;
    cudaMallocManaged(&changes, sizeof(int));
    _iterations_count = 0;
    do
    {
        changes[0] = 0;

        auto edge_op_push = [outgoing_weights, _distances, changes] __device__(int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos){
            _TEdgeWeight weight = outgoing_weights[global_edge_pos];
            _TEdgeWeight src_weight = __ldg(&_distances[src_id]);
            _TEdgeWeight dst_weight = __ldg(&_distances[dst_id]);

            if(dst_weight > src_weight + weight)
            {
                _distances[dst_id] = src_weight + weight;
                changes[0] = 1;
            }
        };

        auto edge_op_pull = [outgoing_weights, _distances, changes] __device__(int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos){
            _TEdgeWeight weight = outgoing_weights[global_edge_pos];
            _TEdgeWeight src_weight = __ldg(&_distances[src_id]);
            _TEdgeWeight dst_weight = __ldg(&_distances[dst_id]);

            if(src_weight > dst_weight + weight)
            {
                _distances[src_id] = dst_weight + weight;
                changes[0] = 1;
            }
        };

        if(_traversal_direction == PUSH_TRAVERSAL)
            graph_API.advance(_graph, frontier, edge_op_push);
        else if(_traversal_direction == PULL_TRAVERSAL)
            graph_API.advance(_graph, frontier, edge_op_pull);
        _iterations_count++;
    }
    while(changes[0] > 0);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void gpu_dijkstra_partial_active_wrapper(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                         _TEdgeWeight *_distances,
                                         int _source_vertex,
                                         int &_iterations_count)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    GraphPrimitivesGPU graph_API;

    char *was_updated;
    cudaMalloc((void**)&was_updated, vertices_count*sizeof(char));

    FrontierGPU frontier(vertices_count);
    frontier.set_all_active();

    auto init_op = [_distances, _source_vertex] __device__ (int src_id, int connections_count) {
        if(src_id == _source_vertex)
            _distances[_source_vertex] = 0;
        else
            _distances[src_id] = FLT_MAX;
    };

    graph_API.compute(_graph, frontier, init_op);

    auto edge_op = [outgoing_weights, _distances, was_updated] __device__(int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos){
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

    auto initial_frontier_condition = [_source_vertex] __device__(int src_id)->int{
        if(src_id == _source_vertex)
            return IN_FRONTIER_FLAG;
        else
            return NOT_IN_FRONTIER_FLAG;
    };

    auto frontier_condition = [was_updated] __device__(int src_id)->int{
        if(was_updated[src_id] > 0)
            return IN_FRONTIER_FLAG;
        else
            return NOT_IN_FRONTIER_FLAG;
    };

    graph_API.generate_new_frontier(_graph, frontier, initial_frontier_condition);

    while(frontier.size() > 0)
    {
        cudaMemset(was_updated, 0, sizeof(char) * vertices_count);
        graph_API.advance(_graph, frontier, edge_op);
        graph_API.generate_new_frontier(_graph, frontier, frontier_condition);
        _iterations_count++;
    }

    cudaFree(was_updated);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template void gpu_dijkstra_all_active_wrapper<int, float>(ExtendedCSRGraph<int, float> &_graph, float *_distances, int _source_vertex,
                                                          int &_iterations_count, TraversalDirection _traversal_direction);
template void gpu_dijkstra_partial_active_wrapper<int, float>(ExtendedCSRGraph<int, float> &_graph, float *_distances, int _source_vertex,
                                                              int &_iterations_count);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
