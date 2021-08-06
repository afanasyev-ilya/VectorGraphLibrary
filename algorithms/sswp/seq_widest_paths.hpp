#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void SSWP::seq_dijkstra(VGL_Graph &_graph,
                        EdgesArray<_T> &_edges_capacities,
                        VerticesArray<_T> &_widths,
                        int _source_vertex)
{
    VectorCSRGraph *outgoing_graph_ptr = _graph.get_outgoing_data();
    LOAD_VECTOR_CSR_GRAPH_DATA((*outgoing_graph_ptr));

    _source_vertex = _graph.reorder(_source_vertex, ORIGINAL, SCATTER);

    for(int i = 0; i < vertices_count; i++)
    {
        _widths[i] = 0.0;
    }
    _widths[_source_vertex] = FLT_MAX;

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

        container.pop();

        long long edge_start = vertex_pointers[src_id];
        int connections_count = vertex_pointers[src_id + 1] - vertex_pointers[src_id];

        for(int edge_pos = 0; edge_pos < connections_count; edge_pos++)
        {
            int dst_id = adjacent_ids[edge_start + edge_pos];
            _T weight = _edges_capacities[edge_start + edge_pos];

            // Finding the widest distance to the vertex
            // using current_source vertex's widest distance
            // and its widest distance so far
            _T distance = max(_widths[dst_id], min(_widths[src_id], weight));

            // Relaxation of edge and adding into Priority Queue
            if (distance > _widths[dst_id])
            {
                // Updating bottle-neck distance
                _widths[dst_id] = distance;

                // Adding the relaxed edge in the prority queue
                container.push(make_pair(distance, dst_id));
            }
        }
    }
    tm.end();


    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    performance_stats.print_algorithm_performance_stats("SSWP (SEQ)", tm.get_time(), _graph.get_edges_count());
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

