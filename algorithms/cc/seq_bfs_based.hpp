#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void CC::seq_bfs_based(VectCSRGraph &_graph, VerticesArray<_T> &_components)
{
    Timer tm;
    tm.start();

    UndirectedCSRGraph *outgoing_graph_ptr = _graph.get_outgoing_graph_ptr();
    LOAD_UNDIRECTED_CSR_GRAPH_DATA((*outgoing_graph_ptr));

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

                const long long edge_start = vertex_pointers[s];
                const int connections_count = vertex_pointers[s + 1] - vertex_pointers[s];

                for(int edge_pos = 0; edge_pos < connections_count; edge_pos++)
                {
                    long long int global_edge_pos = edge_start + edge_pos;
                    int dst_id = adjacent_ids[global_edge_pos];
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
    PerformanceStats::print_algorithm_performance_stats("CC (BFS-based, Sequential)", tm.get_time(), _graph.get_edges_count());
    print_component_sizes(_components);
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
