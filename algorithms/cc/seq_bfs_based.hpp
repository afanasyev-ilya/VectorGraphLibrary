#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void CC::seq_bfs_based(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                       int *_components)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    BFS<int, float> bfs_operation(_graph);

    for(int v = 0; v < vertices_count; v++)
    {
        _components[v] = COMPONENT_UNSET;
    }

    int *bfs_levels;
    MemoryAPI::allocate_array(&bfs_levels, vertices_count);

    int current_component_num = FIRST_COMPONENT;
    for(int v = 0; v < vertices_count; v++)
    {
        if(_components[v] == COMPONENT_UNSET)
        {
            int connections_count = outgoing_ptrs[v + 1] - outgoing_ptrs[v];

            if(connections_count >= 1)
            {
                bfs_operation.seq_top_down(_graph, bfs_levels, v);

                #pragma omp parallel for
                for(int i = 0; i < vertices_count; i++)
                {
                    if(bfs_levels[i] > 0)
                    {
                        _components[i] = current_component_num;
                    }
                }
            }
            else
            {
                _components[v] = current_component_num;
            }

            current_component_num++;
        }
    }

    MemoryAPI::free_array(bfs_levels);

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    PerformanceStats::component_stats(_components, vertices_count);
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
