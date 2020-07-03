#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
bool seq_bfs(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int _source, int _sink, int *_parents)
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

        const long long edge_start = outgoing_ptrs[src_id];
        const int connections_count = outgoing_ptrs[src_id + 1] - outgoing_ptrs[src_id];

        for(int edge_pos = 0; edge_pos < connections_count; edge_pos++)
        {
            long long int global_edge_pos = edge_start + edge_pos;
            int dst_id = outgoing_ids[global_edge_pos];
            _TEdgeWeight weight = outgoing_weights[global_edge_pos];

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

template <typename _TVertexValue, typename _TEdgeWeight>
_TEdgeWeight get_weight(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int _src_id, int _dst_id)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    const long long edge_start = outgoing_ptrs[_src_id];
    const int connections_count = outgoing_ptrs[_src_id + 1] - outgoing_ptrs[_src_id];

    for (int edge_pos = 0; edge_pos < connections_count; edge_pos++)
    {
        long long int global_edge_pos = edge_start + edge_pos;
        int dst_id = outgoing_ids[global_edge_pos];

        if (_dst_id == dst_id)
            return outgoing_weights[global_edge_pos];
    }
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void plus_weight(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int _src_id, int _dst_id, _TEdgeWeight update_val)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    const long long edge_start = outgoing_ptrs[_src_id];
    const int connections_count = outgoing_ptrs[_src_id + 1] - outgoing_ptrs[_src_id];

    for (int edge_pos = 0; edge_pos < connections_count; edge_pos++)
    {
        long long int global_edge_pos = edge_start + edge_pos;
        int dst_id = outgoing_ids[global_edge_pos];

        if (_dst_id == dst_id)
            outgoing_weights[global_edge_pos] += update_val;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void minus_weight(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int _src_id, int _dst_id, _TEdgeWeight update_val)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    const long long edge_start = outgoing_ptrs[_src_id];
    const int connections_count = outgoing_ptrs[_src_id + 1] - outgoing_ptrs[_src_id];

    for (int edge_pos = 0; edge_pos < connections_count; edge_pos++)
    {
        long long int global_edge_pos = edge_start + edge_pos;
        int dst_id = outgoing_ids[global_edge_pos];

        if (_dst_id == dst_id)
            outgoing_weights[global_edge_pos] -= update_val;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
_TEdgeWeight MF::seq_ford_fulkerson(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                    int _source, int _sink)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    int *parents;
    MemoryAPI::allocate_array(&parents, vertices_count);

    for(int i = 0; i < edges_count; i++)
    {
        outgoing_weights[i] = 10;
    }

    int max_flow = 0;  // There is no flow initially

    // Augment the flow while tere is path from source to sink
    int it = 0;
    while (seq_bfs(_graph, _source, _sink, parents))
    {
        // Find minimum residual capacity of the edges along the
        // path filled by BFS. Or we can say find the maximum flow
        // through the path found.
        int path_flow = INT_MAX;
        int path_length = 0;
        for (int v = _sink; v != _source; v = parents[v])
        {
            int u = parents[v];
            int current_weight = get_weight(_graph, u, v);
            path_flow = min(path_flow, current_weight);
        }

        /// update residual capacities of the edges and reverse edges
        // along the path
        for (int v = _sink; v != _source; v = parents[v])
        {
            int u = parents[v];
            minus_weight(_graph, u, v, path_flow);
            plus_weight(_graph, v, u, path_flow);
        }

        // Add path flow to overall flow
        max_flow += path_flow;
    }

    MemoryAPI::free_array(parents);

    // Return the overall flow
    return max_flow;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
