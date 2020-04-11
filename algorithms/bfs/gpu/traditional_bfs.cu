#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../../../graph_processing_API/gpu/cuda_API_include.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace std;

#define FIRST_LEVEL_VERTEX 1
#define UNVISITED_VERTEX -1

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void top_down_wrapper(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                      int *_levels,
                      int _source_vertex,
                      int &_iterations_count)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);
    GraphPrimitivesGPU graph_API;
    FrontierGPU frontier(_graph.get_vertices_count());

    frontier.set_all_active();

    auto init_levels = [_levels, _source_vertex] __device__ (int src_id, int connections_count)
    {
        if(src_id == _source_vertex)
            _levels[_source_vertex] = FIRST_LEVEL_VERTEX;
        else
            _levels[src_id] = UNVISITED_VERTEX;
    };
    graph_API.compute(_graph, frontier, init_levels);

    auto on_first_level = [_levels] __device__ (int src_id)->int
    {
        if(_levels[src_id] == FIRST_LEVEL_VERTEX)
            return IN_FRONTIER_FLAG;
        else
            return NOT_IN_FRONTIER_FLAG;
    };
    graph_API.generate_new_frontier(_graph, frontier, on_first_level);

    int current_level = FIRST_LEVEL_VERTEX;
    while(frontier.size() > 0)
    {
        auto edge_op = [_levels, current_level] __device__ (int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos)
        {
            int src_level = _levels[src_id];
            int dst_level = _levels[dst_id];
            if((src_level == current_level) && (dst_level == UNVISITED_VERTEX))
            {
                _levels[dst_id] = current_level + 1;
            }
        };

        graph_API.advance(_graph, frontier, edge_op);

        auto on_next_level = [_levels, current_level] __device__ (int src_id)->int
        {
            if(_levels[src_id] == (current_level + 1))
                return IN_FRONTIER_FLAG;
            else
                return NOT_IN_FRONTIER_FLAG;
        };

        graph_API.generate_new_frontier(_graph, frontier, on_next_level);

        current_level++;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template void top_down_wrapper<int, float>(ExtendedCSRGraph<int, float> &_graph, int *_levels,
                                           int _source_vertex, int &_iterations_count);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

