#pragma once

#include <functional>
#include <queue>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void SSSP::seq_dijkstra(VectCSRGraph &_graph, EdgesArray_Vect<_T> &_weights, VerticesArray<_T> &_distances,
                        int _source_vertex)
{
    VectorCSRGraph *outgoing_graph_ptr = _graph.get_outgoing_graph_ptr();
    LOAD_VECTOR_CSR_GRAPH_DATA((*outgoing_graph_ptr));

    _source_vertex = _graph.reorder(_source_vertex, ORIGINAL, SCATTER);

    Timer tm;
    tm.start();

    // Create a priority queue to store vertices that
    // are being preprocessed.
    typedef pair<_T, int> iPair;
    std::priority_queue< iPair, vector <iPair> , greater<iPair> > pq;

    // Create a vector for distances and initialize all
    // distances as infinite (INF)
    _T inf_val = std::numeric_limits<_T>::max() - MAX_WEIGHT;
    for(int i = 0; i < vertices_count; i++)
        _distances[i] = inf_val;

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
        int src_id = pq.top().second;
        pq.pop();

        const long long edge_start = vertex_pointers[src_id];
        const int connections_count = vertex_pointers[src_id + 1] - vertex_pointers[src_id];

        for(int edge_pos = 0; edge_pos < connections_count; edge_pos++)
        {
            long long int global_edge_pos = edge_start + edge_pos;
            int dst_id = adjacent_ids[global_edge_pos];
            _T weight = _weights[global_edge_pos];

            if (_distances[dst_id] > _distances[src_id] + weight)
            {
                // Updating distance of dst_id
                _distances[dst_id] = _distances[src_id] + weight;

                pq.push(make_pair(_distances[dst_id], dst_id));
            }
        }
    }
    tm.end();


    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    performance_stats.print_algorithm_performance_stats("SSSP (Dijkstra, Sequential)", tm.get_time(), _graph.get_edges_count());
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
