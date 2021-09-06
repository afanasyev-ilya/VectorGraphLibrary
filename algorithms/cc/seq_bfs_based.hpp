#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
double CC::seq_bfs_based(VGL_Graph &_graph, VerticesArray<_T> &_components)
{
    Timer tm;
    tm.start();

    int vertices_count = _graph.get_vertices_count();

    for(int v = 0; v < vertices_count; v++)
    {
        _components[v] = COMPONENT_UNSET;
    }

    int current_component_num = FIRST_COMPONENT;
    for(int current_vertex = 0; current_vertex < vertices_count; current_vertex++)
    {
        if(_components[current_vertex] == COMPONENT_UNSET)
        {
            int source_vertex = current_vertex;
            list<int> queue;
            _components[source_vertex] = current_component_num;
            queue.push_back(source_vertex);

            while(!queue.empty())
            {
                int s = queue.front();
                queue.pop_front();

                const int connections_count = _graph.get_outgoing_connections_count(s);

                for(int edge_pos = 0; edge_pos < connections_count; edge_pos++)
                {
                    int dst_id = _graph.get_outgoing_edge_dst(s, edge_pos);
                    if(_components[dst_id] == COMPONENT_UNSET)
                    {
                        _components[dst_id] = current_component_num;
                        queue.push_back(dst_id);
                    }
                }
            }

            current_component_num++;
        }
    }
    tm.end();

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    performance_stats.print_algorithm_performance_stats("CC (BFS-based, Sequential)", tm.get_time(), _graph.get_edges_count());
    print_component_sizes(_components);
    #endif

    return performance_stats.get_algorithm_performance(tm.get_time(), _graph.get_edges_count());
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
