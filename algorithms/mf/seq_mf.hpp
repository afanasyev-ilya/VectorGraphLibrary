#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


bool MF::seq_bfs(ExtendedCSRGraph &_graph, int _source, int _sink, int *_parents)
{
    // Create a visited array and mark all vertices as not visited
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

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

        const long long edge_start = vertex_pointers[src_id];
        const int connections_count = vertex_pointers[src_id + 1] - vertex_pointers[src_id];

        for(int edge_pos = 0; edge_pos < connections_count; edge_pos++)
        {
            long long int global_edge_pos = edge_start + edge_pos;
            int dst_id = adjacent_ids[global_edge_pos];
            _TEdgeWeight weight = adjacent_weights[global_edge_pos];

            if (visited[dst_id] == false && weight > 0)
            {
                q.push(dst_id);
                _parents[dst_id] = src_id;
                visited[dst_id] = true;
            }
        }
    }

    // If we reached sink in BFS starting from source, then return
    // true, else false
    return (visited[_sink] == true);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


_TEdgeWeight MF::get_flow(ExtendedCSRGraph &_graph, int _src_id, int _dst_id)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    const long long edge_start = vertex_pointers[_src_id];
    const int connections_count = vertex_pointers[_src_id + 1] - vertex_pointers[_src_id];

    for (int edge_pos = 0; edge_pos < connections_count; edge_pos++)
    {
        long long int global_edge_pos = edge_start + edge_pos;
        int dst_id = adjacent_ids[global_edge_pos];

        if (_dst_id == dst_id)
            return adjacent_weights[global_edge_pos];
    }
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void MF::add_flow(ExtendedCSRGraph &_graph, int _src_id, int _dst_id, _TEdgeWeight update_val)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    const long long edge_start = vertex_pointers[_src_id];
    const int connections_count = vertex_pointers[_src_id + 1] - vertex_pointers[_src_id];

    for (int edge_pos = 0; edge_pos < connections_count; edge_pos++)
    {
        long long int global_edge_pos = edge_start + edge_pos;
        int dst_id = adjacent_ids[global_edge_pos];

        if (_dst_id == dst_id)
            adjacent_weights[global_edge_pos] += update_val;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void MF::subtract_flow(ExtendedCSRGraph &_graph, int _src_id, int _dst_id, _TEdgeWeight update_val)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    const long long edge_start = vertex_pointers[_src_id];
    const int connections_count = vertex_pointers[_src_id + 1] - vertex_pointers[_src_id];

    for (int edge_pos = 0; edge_pos < connections_count; edge_pos++)
    {
        long long int global_edge_pos = edge_start + edge_pos;
        int dst_id = adjacent_ids[global_edge_pos];

        if (_dst_id == dst_id)
            adjacent_weights[global_edge_pos] -= update_val;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


_TEdgeWeight MF::seq_ford_fulkerson(ExtendedCSRGraph &_graph,
                                    int _source, int _sink)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    int *parents;
    MemoryAPI::allocate_array(&parents, vertices_count);

    for(int i = 0; i < edges_count; i++)
    {
        adjacent_weights[i] = 10;
    }

    int max_flow = 0;  // There is no flow initially

    // Augment the flow while tere is path from source to sink
    int it = 0;
    while (seq_bfs(_graph, _source, _sink, parents))
    {
        // Find minimum residual capacity of the edges along the
        // path filled by BFS. Or we can say find the maximum flow
        // through the path found.
        int path_flow = std::numeric_limits<std::int32_t>::max();
        int path_length = 0;
        for (int v = _sink; v != _source; v = parents[v])
        {
            int u = parents[v];
            int current_weight = get_flow(_graph, u, v);
            path_flow = min(path_flow, current_weight);
        }

        /// update residual capacities of the edges and reverse edges
        // along the path
        for (int v = _sink; v != _source; v = parents[v])
        {
            int u = parents[v];
            subtract_flow(_graph, u, v, path_flow);
            add_flow(_graph, v, u, path_flow);
        }

        // Add path flow to overall flow
        max_flow += path_flow;
    }

    MemoryAPI::free_array(parents);

    // Return the overall flow
    return max_flow;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
