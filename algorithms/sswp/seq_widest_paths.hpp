#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void SSWP::seq_dijkstra(VectCSRGraph &_graph,
                        EdgesArray_Vect<_T> &_edges_capacities,
                        VerticesArray<_T> &_widths,
                        int _source_vertex)
{
    UndirectedCSRGraph *outgoing_graph_ptr = _graph.get_outgoing_graph_ptr();
    LOAD_UNDIRECTED_CSR_GRAPH_DATA((*outgoing_graph_ptr));

    _source_vertex = _graph.reorder(_source_vertex, ORIGINAL, SCATTER);

    _T inf_val = std::numeric_limits<_T>::max() - MAX_WEIGHT;
    for(int i = 0; i < vertices_count; i++)
    {
        _widths[i] = 0.0;
    }
    _widths[_source_vertex] = inf_val;

    // Use of Minimum Priority Queue to keep track minimum
    // widest distance vertex so far in the algorithm
    priority_queue<pair<float, int>, vector<pair<float, int> >, greater<pair<float, int> > > container;

    container.push(make_pair(inf_val, _source_vertex));

    while (container.empty() == false)
    {
        pair<float, int> temp = container.top();

        int src_id = temp.second;

        container.pop();

        long long edge_start = vertex_pointers[src_id];
        int connections_count = vertex_pointers[src_id + 1] - vertex_pointers[src_id];

        for(int edge_pos = 0; edge_pos < connections_count; edge_pos++)
        {
            /*int dst_id = adjacent_ids[edge_start + edge_pos];
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
            }*/

            int dst_id = adjacent_ids[edge_start + edge_pos];
            _T edge_width = _edges_capacities[edge_start + edge_pos];
            _T new_width = vect_min(_widths[src_id], edge_width);

            if(_widths[dst_id] < new_width)
            {
                _widths[dst_id] = new_width;
                container.push(make_pair(new_width, dst_id));
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

