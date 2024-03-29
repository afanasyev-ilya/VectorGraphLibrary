#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
double SSWP::seq_dijkstra(VGL_Graph &_graph,
                          EdgesArray<_T> &_edges_capacities,
                          VerticesArray<_T> &_widths,
                          int _source_vertex)
{
    int vertices_count = _graph.get_vertices_count();

    _source_vertex = _graph.reorder(_source_vertex, ORIGINAL, SCATTER);

    for(int i = 0; i < vertices_count; i++)
    {
        _widths[i] = 0.0;
    }
    _widths[_source_vertex] = std::numeric_limits<_T>::max();

    // Use of Minimum Priority Queue to keep track minimum
    // widest distance vertex so far in the algorithm
    priority_queue<pair<_T, int>, vector<pair<_T, int> >, greater<pair<_T, int> > > container;

    container.push(make_pair(0, _source_vertex));

    Timer tm;
    tm.start();
    while(!container.empty())
    {
        pair<_T, int> temp = container.top();

        int src_id = temp.second;
        _T src_width = temp.first;

        container.pop();

        int connections_count = _graph.get_outgoing_connections_count(src_id);

        for(int edge_pos = 0; edge_pos < connections_count; edge_pos++)
        {
            int dst_id = _graph.get_outgoing_edge_dst(src_id, edge_pos);
            _T edge_width = _edges_capacities[_graph.get_outgoing_edges_array_index(src_id, edge_pos)];

            _T distance = max(_widths[dst_id], min(_widths[src_id], edge_width));

            // Relaxation of edge and adding into Priority Queue
            if (distance > _widths[dst_id])
            {
                // Updating bottle-neck distance
                _widths[dst_id] = distance;

                // Adding the relaxed edge in the priority queue
                container.push(make_pair(distance, dst_id));
            }
        }
    }
    tm.end();

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    performance_stats.print_algorithm_performance_stats("SSWP (SEQ)", tm.get_time(), _graph.get_edges_count());
    #endif

    return performance_stats.get_algorithm_performance(tm.get_time(), _graph.get_edges_count());
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

