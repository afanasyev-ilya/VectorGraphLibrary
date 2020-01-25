#pragma once

#include <functional>
#include <queue>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef pair<float, int> iPair;

template <typename _TVertexValue, typename _TEdgeWeight>
void ShortestPaths<_TVertexValue, _TEdgeWeight>::seq_dijkstra(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                              int _source_vertex,
                                                              _TEdgeWeight *_distances)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    double t1 = omp_get_wtime();

    // Create a priority queue to store vertices that
    // are being preprocessed. This is weird syntax in C++.
    std::priority_queue< iPair, vector <iPair> , greater<iPair> > pq;

    // Create a vector for distances and initialize all
    // distances as infinite (INF)
    for(int i = 0; i < vertices_count; i++)
        _distances[i] = FLT_MAX;

    // Insert source itself in priority queue and initialize
    // its distance as 0.
    pq.push(make_pair(0, _source_vertex));
    _distances[_source_vertex] = 0;

    /* Looping till priority queue becomes empty (or all
      distances are not finalized) */
    while (!pq.empty())
    {
        // The first vertex in pair is the minimum distance
        // vertex, extract it from priority queue.
        // vertex label is stored in second of pair (it
        // has to be done this way to keep the vertices
        // sorted distance (distance must be first item
        // in pair)
        int u = pq.top().second;
        pq.pop();

        const long long edge_start = outgoing_ptrs[u];
        const int connections_count = outgoing_ptrs[u + 1] - outgoing_ptrs[u];

        for(register int edge_pos = 0; edge_pos < connections_count; edge_pos++)
        {
            long long int global_edge_pos = edge_start + edge_pos;
            int v = outgoing_ids[global_edge_pos];
            _TEdgeWeight weight = outgoing_weights[global_edge_pos];

            if (_distances[v] > _distances[u] + weight)
            {
                // Updating distance of v
                _distances[v] = _distances[u] + weight;
                pq.push(make_pair(_distances[v], v));
            }
        }
    }

    double t2 = omp_get_wtime();

    print_performance_stats(edges_count, 1, t2 - t1);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
