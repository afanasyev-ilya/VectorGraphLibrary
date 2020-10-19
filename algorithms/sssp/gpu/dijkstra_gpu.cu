#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../../../graph_processing_API/gpu/cuda_API_include.h"
#define INT_ELEMENTS_PER_EDGE 5.0
#define __PRINT_API_PERFORMANCE_STATS__

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void gpu_dijkstra_all_active_push_wrapper(VectCSRGraph &_graph,
                                          EdgesArray<_T> &_weights,
                                          VerticesArray<_T> &_distances,
                                          int _source_vertex,
                                          int &_iterations_count)
{
    GraphAbstractionsGPU graph_API(_graph, SCATTER);
    FrontierGPU frontier(_graph, SCATTER);
    //graph_API.change_traversal_direction(SCATTER, _distances, frontier);

    _T inf_val = std::numeric_limits<_T>::max() - MAX_WEIGHT;
    auto init_op = [_distances, _source_vertex, inf_val] __device__  (int src_id, int connections_count, int vector_index) {
        if(src_id == _source_vertex)
            _distances[_source_vertex] = 0;
        else
            _distances[src_id] = inf_val;
    };
    frontier.set_all_active();
    graph_API.compute(_graph, frontier, init_op);

    int *changes;
    MemoryAPI::allocate_array(&changes, 1);
    _iterations_count = 0;
    do
    {
        changes[0] = 0;

        auto edge_op = [_weights, _distances, changes] __device__(int src_id, int dst_id, int local_edge_pos,
                              long long int global_edge_pos, int vector_index){
            _T weight = _weights[global_edge_pos];
            _T src_weight = __ldg(&_distances[src_id]);
            _T dst_weight = __ldg(&_distances[dst_id]);

            if(dst_weight > src_weight + weight)
            {
                _distances[dst_id] = src_weight + weight;
                changes[0] = 1;
            }
        };

        graph_API.scatter(_graph, frontier, edge_op);

        _iterations_count++;
    }
    while(changes[0] > 0);

    MemoryAPI::free_array(changes);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void gpu_dijkstra_all_active_pull_wrapper(VectCSRGraph &_graph,
                                          EdgesArray<_T> &_weights,
                                          VerticesArray<_T> &_distances,
                                          int _source_vertex,
                                          int &_iterations_count)
{
    GraphAbstractionsGPU graph_API(_graph, GATHER);
    FrontierGPU frontier(_graph, GATHER);
    //graph_API.change_traversal_direction(GATHER, _distances, frontier);

    _T inf_val = std::numeric_limits<_T>::max() - MAX_WEIGHT;
    auto init_op = [_distances, _source_vertex, inf_val] __device__  (int src_id, int connections_count, int vector_index) {
        if(src_id == _source_vertex)
            _distances[_source_vertex] = 0;
        else
            _distances[src_id] = inf_val;
    };
    frontier.set_all_active();
    graph_API.compute(_graph, frontier, init_op);

    int *changes;
    MemoryAPI::allocate_array(&changes, 1);
    _iterations_count = 0;
    do
    {
        changes[0] = 0;

        auto edge_op = [_weights, _distances, changes] __device__(int src_id, int dst_id, int local_edge_pos,
                long long int global_edge_pos, int vector_index){
            _T weight = _weights[global_edge_pos];
            _T src_weight = __ldg(&_distances[src_id]);
            _T dst_weight = __ldg(&_distances[dst_id]);

            if(src_weight > dst_weight + weight)
            {
                _distances[src_id] = dst_weight + weight;
                changes[0] = 1;
            }
        };

        graph_API.gather(_graph, frontier, edge_op);

        _iterations_count++;
    }
    while(changes[0] > 0);

    MemoryAPI::free_array(changes);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void gpu_dijkstra_partial_active_wrapper(VectCSRGraph &_graph,
                                         EdgesArray<_T> &_weights,
                                         VerticesArray<_T> &_distances,
                                         int _source_vertex,
                                         int &_iterations_count)
{
    GraphAbstractionsGPU graph_API(_graph, SCATTER);
    FrontierGPU frontier(_graph, SCATTER);
    VerticesArray<char> was_updated(_graph, SCATTER);
    //graph_API.change_traversal_direction(SCATTER, _distances, frontier, was_updated);

    _T inf_val = std::numeric_limits<_T>::max() - MAX_WEIGHT;
    auto init_op = [_distances, _source_vertex, inf_val] __device__  (int src_id, int connections_count, int vector_index) {
        if(src_id == _source_vertex)
            _distances[_source_vertex] = 0;
        else
            _distances[src_id] = inf_val;
    };
    frontier.set_all_active();
    graph_API.compute(_graph, frontier, init_op);

    auto edge_op = [_weights, _distances, was_updated] __device__(int src_id, int dst_id, int local_edge_pos,
            long long int global_edge_pos, int vector_index) {
        _T weight = _weights[global_edge_pos];
        _T src_weight = __ldg(&_distances[src_id]);
        _T dst_weight = __ldg(&_distances[dst_id]);

        if(dst_weight > src_weight + weight)
        {
            _distances[dst_id] = src_weight + weight;
            was_updated[dst_id] = 1;
            was_updated[src_id] = 1;
        }
    };

    auto frontier_condition = [was_updated] __device__(int src_id, int connections_count)->int{
        if(was_updated[src_id] > 0)
            return IN_FRONTIER_FLAG;
        else
            return NOT_IN_FRONTIER_FLAG;
    };

    frontier.clear();
    frontier.add_vertex(_source_vertex);

    while(frontier.size() > 0)
    {
        cudaMemset(was_updated.get_ptr(), 0, sizeof(char) * _graph.get_vertices_count());
        graph_API.scatter(_graph, frontier, edge_op);
        graph_API.generate_new_frontier(_graph, frontier, frontier_condition);
        _iterations_count++;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template void gpu_dijkstra_all_active_push_wrapper<int>(VectCSRGraph &_graph, EdgesArray<int> &_weights,
               VerticesArray<int> &_distances, int _source_vertex, int &_iterations_count);
template void gpu_dijkstra_all_active_push_wrapper<float>(VectCSRGraph &_graph, EdgesArray<float> &_weights,
               VerticesArray<float> &_distances, int _source_vertex, int &_iterations_count);

template void gpu_dijkstra_all_active_pull_wrapper<int>(VectCSRGraph &_graph, EdgesArray<int> &_weights,
                                                        VerticesArray<int> &_distances, int _source_vertex, int &_iterations_count);
template void gpu_dijkstra_all_active_pull_wrapper<float>(VectCSRGraph &_graph, EdgesArray<float> &_weights,
                                                          VerticesArray<float> &_distances, int _source_vertex, int &_iterations_count);

template void gpu_dijkstra_partial_active_wrapper<int>(VectCSRGraph &_graph, EdgesArray<int> &_weights,
                                                       VerticesArray<int> &_distances, int _source_vertex, int &_iterations_count);
template void gpu_dijkstra_partial_active_wrapper<float>(VectCSRGraph &_graph, EdgesArray<float> &_weights,
                                                         VerticesArray<float> &_distances, int _source_vertex, int &_iterations_count);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
