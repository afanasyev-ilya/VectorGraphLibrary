#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int calculate_remaining_count(int *_components, int _vertices_count)
{
    int count = 0;
    #pragma omp parallel for schedule(static) reduction(+: count)
    for(int src_id = 0; src_id < _vertices_count; src_id++)
    {
        if(_components[src_id] == COMPONENT_UNSET)
            count++;
    }
    return count;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
void CC::nec_bfs_based(UndirectedCSRGraph &_graph,
                       int *_components)
{
    LOAD_UNDIRECTED_CSR_GRAPH_DATA(_graph);

    int *bfs_levels;
    BFS<_TVertexValue,_TEdgeWeight> bfs_operation(_graph);
    bfs_operation.allocate_result_memory(_graph.get_vertices_count(), &bfs_levels);

    frontier.set_all_active();
    auto init_components_op = [_components] (int src_id, int connections_count, int vector_index)
    {
        _components[src_id] = COMPONENT_UNSET;
    };
    graph_API.compute(_graph, frontier, init_components_op);

    auto all_active = [] (int src_id)->int
    {
        return IN_FRONTIER_FLAG;
    };
    graph_API.filter(_graph, frontier, all_active);

    auto remove_zero_nodes_op = [_components, vertices_count] (int src_id, int connections_count, int vector_index)
    {
        if((connections_count == 0) && (src_id < vertices_count))
            _components[src_id] = SINGLE_VERTEX_COMPONENT;

        if((connections_count == 1) && (src_id < vertices_count))
            _components[src_id] = DUO_VERTEX_COMPONENT;
    };
    graph_API.compute(_graph, frontier, remove_zero_nodes_op);

    double t1 = omp_get_wtime();
    int current_component = FIRST_COMPONENT;
    int iterations_count = 0;
    for(int v = 0; v < vertices_count; v++)
    {
        if(_components[v] == COMPONENT_UNSET)
        {
            bfs_operation.nec_top_down(_graph, bfs_levels, v);
            auto copy_levels_to_components_op = [_components, bfs_levels, current_component] (int src_id, int connections_count, int vector_index)
            {
                if(bfs_levels[src_id] > 0)
                {
                    _components[src_id] = current_component;
                }
            };
            graph_API.compute(_graph, frontier, copy_levels_to_components_op);
            current_component++;
            iterations_count++;
            if(v == 0)
            {
                int remaining_count = calculate_remaining_count(_components, vertices_count);
                if(remaining_count == 0)
                    break;
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
            _components[adjacent_ids[vertex_pointers[v]]] = current_component;
            current_component++;
        }
    }
    double t2 = omp_get_wtime();

    performance = edges_count / ((t2 - t1)*1e6);

    bfs_operation.free_result_memory(bfs_levels);

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    PerformanceStats::print_algorithm_performance_stats("nec bfs based", t2 - t1, edges_count, iterations_count);
    PerformanceStats::component_stats(_components, vertices_count);
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

