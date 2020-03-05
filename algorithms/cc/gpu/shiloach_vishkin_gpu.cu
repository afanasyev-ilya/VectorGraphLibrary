#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../../../graph_processing_API/gpu/cuda_API_include.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void shiloach_vishkin_wrapper(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                              int *_components,
                              int &_iterations_count)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    GraphPrimitivesGPU graph_API;
    FrontierGPU frontier(_graph.get_vertices_count());

    auto init_components_op = [_components] __device__ (int src_id)
    {
        _components[src_id] = src_id;
    };
    graph_API.compute(init_components_op, vertices_count);

    auto all_active = [] __device__ (int src_id)->int
    {
        return 1;
    };
    frontier.filter(_graph, all_active);

    int hook_changes = 1, jump_changes = 1;
    int iteration = 0;
    while(hook_changes)
    {
        hook_changes = 0;


    }

    _iterations_count = iteration;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template void shiloach_vishkin_wrapper<int, float>(ExtendedCSRGraph<int, float> &_graph, int *_components,
                                                   int &_iterations_count);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////