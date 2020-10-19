#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../../../architectures.h"
#include "../../../graph_processing_API/gpu/cuda_API_include.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void shiloach_vishkin_wrapper(UndirectedCSRGraph &_graph,
                              int *_components,
                              int &_iterations_count)
{
    LOAD_UNDIRECTED_CSR_GRAPH_DATA(_graph);

    GraphAbstractionsGPU graph_API;
    FrontierGPU frontier(_graph.get_vertices_count());
    frontier.set_all_active();

    auto init_components_op = [_components] __device__ (int src_id, int position_in_frontier, int connections_count)
    {
        _components[src_id] = src_id;
    };
    graph_API.compute(_graph, frontier, init_components_op);

    int *hook_changes, *jump_changes;
    cudaMallocManaged(&hook_changes, sizeof(int));
    cudaMallocManaged(&jump_changes, sizeof(int));

    _iterations_count = 0;
    do
    {
        hook_changes[0] = 0;

        auto edge_op = [_components, hook_changes, _iterations_count] __device__(int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos, int position_in_frontier)
        {
            int src_val = _components[src_id];
            int dst_val = _components[dst_id];

            if(src_val < dst_val)
            {
                _components[dst_id] = src_val;
                hook_changes[0] = 1;
            }

            if(src_val > dst_val)
            {
                _components[src_id] = dst_val;
                hook_changes[0] = 1;
            }
        };

        graph_API.advance(_graph, frontier, edge_op);

        do
        {
            jump_changes[0] = 0;
            auto jump_op = [_components, jump_changes] __device__(int src_id, int position_in_frontier, int connections_count)
            {
                int src_label = _components[src_id];
                int parent_label = _components[src_label];

                if(src_label != parent_label)
                {
                    _components[src_id] = parent_label;
                    jump_changes[0] = 0;
                }
            };

            graph_API.compute(_graph, frontier, jump_op);
        } while(jump_changes[0] > 0);

        _iterations_count++;
    } while(hook_changes[0] > 0);

    cudaFree(hook_changes);
    cudaFree(jump_changes);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template void shiloach_vishkin_wrapper<int, float>(UndirectedCSRGraph<int, float> &_graph, int *_components,
                                                   int &_iterations_count);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////