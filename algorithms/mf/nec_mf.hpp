#pragma once

#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
bool nec_bfs(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int _source, int _sink, int *_parents)
{
    // Create a visited array and mark all vertices as not visited
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _TVertexValue, typename _TEdgeWeight>
_TEdgeWeight MF::nec_ford_fulkerson(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                    int _source, int _sink)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    int *parents;
    MemoryAPI::allocate_array(&parents, vertices_count);

    #pragma omp parallel for
    for(int i = 0; i < edges_count; i++)
    {
        outgoing_weights[i] = 10;
    }

    int max_flow = 0;

    int it = 0;
    while(true)
    {
        bool sink_reached = nec_bfs(_graph, _source, _sink, parents);

        if(!sink_reached)
            break;

        int path_flow = INT_MAX;
        int path_length = 0;
        for (int v = _sink; v != _source; v = parents[v])
        {
            int u = parents[v];
            int current_weight = get_weight(_graph, u, v);
            path_flow = min(path_flow, current_weight);
        }

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
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
