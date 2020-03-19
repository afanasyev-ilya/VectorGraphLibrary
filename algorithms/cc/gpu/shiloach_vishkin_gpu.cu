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

    auto all_active = [] __device__ (int src_id)->bool
    {
        if(src_id >= 0)
            return true;
    };
    frontier.filter(_graph, all_active);

    int *hook_changes, *jump_changes;
    cudaMallocManaged(&hook_changes, sizeof(int));
    cudaMallocManaged(&jump_changes, sizeof(int));
    hook_changes[0] = 1;
    jump_changes[0] = 0;

    _iterations_count = 0;
    while(hook_changes[0])
    {
        hook_changes[0] = 0;

        auto EMPTY_VERTEX_OP = [] __device__(int src_id, int connections_count){};
        auto edge_op = [_components, hook_changes]__device__(int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos, int connections_count)
        {
            int src_val = _components[src_id];
            int dst_val = _components[dst_id];

            if(src_val < dst_val)
            {
                int dst_dst_val = _components[_components[dst_id]];
                if(dst_val == dst_dst_val)
                {
                    _components[dst_val] = src_val;
                    hook_changes[0] = 0;
                }
            }
        };

        graph_API.advance(_graph, frontier, edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);

        jump_changes[0] = 1;
        while(jump_changes[0])
        {
            jump_changes[0] = 0;

            auto jump_op = [_components, jump_changes] __device__(int src_id)
            {
                int src_val = _components[src_id];
                int src_src_val = _components[src_val];

                if(src_val != src_src_val)
                {
                    _components[src_id] = src_src_val;
                    jump_changes[0] = 1;
                }
            };

            graph_API.compute(jump_op, vertices_count);
        }

        _iterations_count++;
    }

    cudaFree(hook_changes);
    cudaFree(jump_changes);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template void shiloach_vishkin_wrapper<int, float>(ExtendedCSRGraph<int, float> &_graph, int *_components,
                                                   int &_iterations_count);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////