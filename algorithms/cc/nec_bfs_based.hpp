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
template <typename _T>
void CC::nec_bfs_based(VectCSRGraph &_graph, VerticesArray<_T> &_components)
{
    GraphAbstractionsNEC graph_API(_graph, SCATTER);
    FrontierNEC frontier(_graph, SCATTER);
    VerticesArray<_T> bfs_levels(_graph, SCATTER);

    #pragma omp parallel
    {};

    Timer tm;
    tm.start();

    frontier.set_all_active();
    auto init_components_op = [&_components] (int src_id, int connections_count, int vector_index)
    {
        _components[src_id] = COMPONENT_UNSET;
    };
    graph_API.compute(_graph, frontier, init_components_op);

    int vertices_count = _graph.get_vertices_count();
    int current_component_num = FIRST_COMPONENT;
    for(int current_vertex = 0; current_vertex < vertices_count; current_vertex++)
    {
        if(_components[current_vertex] == COMPONENT_UNSET)
        {
            int source_vertex = current_vertex;

            int current_level = COMPONENT_UNSET - 1;
            while(frontier.size() > 0)
            {
                cout << "current level: " << current_level << endl;

                auto edge_op = [&_components, current_level](int src_id, int dst_id, int local_edge_pos,
                        long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
                {
                    int dst_comp = _components[dst_id];
                    if(dst_comp == COMPONENT_UNSET)
                    {
                        _components[dst_id] = current_level - 1;
                    }
                };

                graph_API.scatter(_graph, frontier, edge_op);

                auto on_next_level = [&_components, current_level] (int src_id, int connections_count)->int
                {
                    int result = NOT_IN_FRONTIER_FLAG;
                    if(_components[src_id] == (current_level - 1))
                        result = IN_FRONTIER_FLAG;
                    return result;
                };

                graph_API.generate_new_frontier(_graph, frontier, on_next_level);

                current_level--;
            }

            frontier.set_all_active();
            auto mark_component = [&_components, current_component_num] (int src_id, int connections_count, int vector_index)
            {
                if(_components[src_id] < COMPONENT_UNSET)
                    _components[src_id] = current_component_num;
            };
            graph_API.compute(_graph, frontier, init_components_op);

            current_component_num++;
        }
    }
    tm.end();

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    PerformanceStats::print_algorithm_performance_stats("CC (BFS-based, NEC)", tm.get_time(), _graph.get_edges_count());
    print_component_sizes(_components);
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

