#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
bool MF::seq_bfs(VGL_Graph &_graph, EdgesArray<_T> &_weights, int _source, int _sink, int *_parents)
{
    // Create a visited array and mark all vertices as not visited
    int vertices_count = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count();

    bool *visited = new bool[vertices_count];
    memset(visited, 0, sizeof(bool)*vertices_count);

    // Create a queue, enqueue source vertex and mark source vertex
    // as visited
    queue <int> q;
    q.push(_source);
    visited[_source] = true;
    _parents[_source] = -1;

    // Standard BFS Loop
    while (!q.empty())
    {
        int src_id = q.front();
        q.pop();

        const int connections_count = _graph.get_outgoing_connections_count(src_id);

        for(int edge_pos = 0; edge_pos < connections_count; edge_pos++)
        {
            int dst_id = _graph.get_outgoing_edge_dst(src_id, edge_pos);
            _T weight = _weights[_graph.get_outgoing_edges_array_index(src_id, edge_pos)];

            if (visited[dst_id] == false && weight > 0)
            {
                q.push(dst_id);
                _parents[dst_id] = src_id;
                visited[dst_id] = true;
            }
        }
    }

    // If we reached sink in BFS starting from source, then return true, else false
    return (visited[_sink] == true);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
_T MF::get_flow(VGL_Graph &_graph, EdgesArray<_T> &_weights, int _src_id, int _dst_id)
{
    const int connections_count = _graph.get_outgoing_connections_count(_src_id);

    for (int edge_pos = 0; edge_pos < connections_count; edge_pos++)
    {
        int dst_id = _graph.get_outgoing_edge_dst(_src_id, edge_pos);

        if (_dst_id == dst_id)
            return _weights[_graph.get_outgoing_edges_array_index(_src_id, edge_pos)];
    }
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void MF::add_flow(VGL_Graph &_graph, EdgesArray<_T> &_weights, int _src_id, int _dst_id, _T update_val)
{
    const int connections_count = _graph.get_outgoing_connections_count(_src_id);

    for (int edge_pos = 0; edge_pos < connections_count; edge_pos++)
    {
        int dst_id = _graph.get_outgoing_edge_dst(_src_id, edge_pos);

        if (_dst_id == dst_id)
            _weights[_graph.get_outgoing_edges_array_index(_src_id, edge_pos)] += update_val;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void MF::subtract_flow(VGL_Graph &_graph, EdgesArray<_T> &_weights, int _src_id, int _dst_id, _T update_val)
{
    const int connections_count = _graph.get_outgoing_connections_count(_src_id);

    for (int edge_pos = 0; edge_pos < connections_count; edge_pos++)
    {
        int dst_id = _graph.get_outgoing_edge_dst(_src_id, edge_pos);

        if (_dst_id == dst_id)
            _weights[_graph.get_outgoing_edges_array_index(_src_id, edge_pos)] -= update_val;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
double MF::seq_ford_fulkerson(VGL_Graph &_graph, int _source, int _sink, _T _max_flow)
{
    Timer tm;
    tm.start();

    int vertices_count = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count();

    int *parents;
    MemoryAPI::allocate_array(&parents, vertices_count);

    EdgesArray<_T> flows(_graph);
    flows.set_all_constant(MAX_WEIGHT);

    _max_flow = 0;  // There is no flow initially

    // Augment the flow while tere is path from source to sink
    int it = 0;
    while (seq_bfs(_graph, flows, _source, _sink, parents))
    {
        // Find minimum residual capacity of the edges along the
        // path filled by BFS. Or we can say find the maximum flow
        // through the path found.
        _T path_flow = std::numeric_limits<_T>::max();
        int path_length = 0;
        for (int v = _sink; v != _source; v = parents[v])
        {
            int u = parents[v];
            _T current_weight = get_flow(_graph, flows, u, v);
            path_flow = min(path_flow, current_weight);
            path_length++;
        }
        cout << "path_length: " << path_length << endl;

        /// update residual capacities of the edges and reverse edges
        // along the path
        for (int v = _sink; v != _source; v = parents[v])
        {
            int u = parents[v];
            subtract_flow(_graph, flows, u, v, path_flow);
            add_flow(_graph, flows, v, u, path_flow);
        }

        // Add path flow to overall flow
        _max_flow += path_flow;
    }

    MemoryAPI::free_array(parents);

    tm.end();

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    performance_stats.print_algorithm_performance_stats("MF (Sequential Ford-Fulkerson)", tm.get_time(), _graph.get_edges_count());
    #endif

    return performance_stats.get_algorithm_performance(tm.get_time(), _graph.get_edges_count());

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
