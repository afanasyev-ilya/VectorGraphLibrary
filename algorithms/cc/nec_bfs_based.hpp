#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _TVertexValue, typename _TEdgeWeight>
void CC::nec_bfs_based(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                       int *_components)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    int *bfs_levels;
    BFS<_TVertexValue,_TEdgeWeight> bfs_operation(_graph);
    bfs_operation.allocate_result_memory(_graph.get_vertices_count(), &bfs_levels);

    auto init_components_op = [_components] (int src_id)
    {
        _components[src_id] = COMPONENT_UNSET;
    };
    graph_API.compute(init_components_op, vertices_count);

    auto all_active = [] (int src_id)->int
    {
        return NEC_IN_FRONTIER_FLAG;
    };
    frontier.filter(_graph, all_active);

    auto remove_zero_nodes_op = [_components, vertices_count] (int src_id, int connections_count, DelayedWriteNEC &delayed_write)
    {
        if((connections_count == 0) && (src_id < vertices_count))
            _components[src_id] = SINGLE_VERTEX_COMPONENT;

        if((connections_count == 1) && (src_id < vertices_count))
            _components[src_id] = DUO_VERTEX_COMPONENT;
    };
    graph_API.advance(_graph, frontier, EMPTY_EDGE_OP, remove_zero_nodes_op, EMPTY_VERTEX_OP);

    double t1 = omp_get_wtime();
    int current_component = FIRST_COMPONENT;
    int iterations_count = 0;
    for(int v = 1; v < vertices_count; v++)
    {
        if(_components[v] == COMPONENT_UNSET)
        {
            bfs_operation.nec_direction_optimising(_graph, bfs_levels, v);
            auto copy_levels_to_components_op = [_components, bfs_levels, current_component] (int src_id)
            {
                if(bfs_levels[src_id] > 0)
                {
                    _components[src_id] = current_component;
                }
            };
            graph_API.compute(copy_levels_to_components_op, vertices_count);
            current_component++;
            iterations_count++;
            if(v == 1)
            {
                v = -1;
                continue;
            }
        }
        else if(_components[v] == SINGLE_VERTEX_COMPONENT)
        {
            _components[v] = current_component;
            current_component++;
        }
        else if(_components[v] == DUO_VERTEX_COMPONENT)
        {
            _components[v] = current_component;
            _components[outgoing_ids[outgoing_ptrs[v]]] = current_component;
            current_component++;
        }
    }
    double t2 = omp_get_wtime();

    bfs_operation.free_result_memory(bfs_levels);

    performance_stats("nec bfs based", t2 - t1, edges_count, iterations_count);

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    component_stats(_components, vertices_count);
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

