#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void SSWP::seq_dijkstra(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                        _TEdgeWeight *_widths,
                        int _source_vertex)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    for(int i = 0; i < vertices_count; i++)
    {
        _widths[i] = 0.0;
    }
    _widths[_source_vertex] = FLT_MAX;

    // Use of Minimum Priority Queue to keep track minimum
    // widest distance vertex so far in the algorithm
    priority_queue<pair<float, int>, vector<pair<float, int> >, greater<pair<float, int> > > container;

    container.push(make_pair(0, _source_vertex));

    while (container.empty() == false)
    {
        pair<float, int> temp = container.top();

        int src_id = temp.second;

        container.pop();

        long long edge_start = vertex_pointers[src_id];
        int connections_count = vertex_pointers[src_id + 1] - vertex_pointers[src_id];

        for(int edge_pos = 0; edge_pos < connections_count; edge_pos++)
        {
            int dst_id = adjacent_ids[edge_start + edge_pos];
            _TEdgeWeight weight = adjacent_weights[edge_start + edge_pos];

            // Finding the widest distance to the vertex
            // using current_source vertex's widest distance
            // and its widest distance so far
            float distance = max(_widths[dst_id], min(_widths[src_id], weight));

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
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

