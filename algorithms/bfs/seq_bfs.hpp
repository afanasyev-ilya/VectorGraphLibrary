#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <queue>
#include <list>

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void BFS::seq_top_down(VectCSRGraph &_graph,
                       VerticesArray<_T> &_levels,
                       int _source_vertex)
{
    UndirectedCSRGraph *outgoing_graph_ptr = _graph.get_outgoing_graph_ptr();
    LOAD_UNDIRECTED_CSR_GRAPH_DATA((*outgoing_graph_ptr));

    _source_vertex = _graph.reorder(_source_vertex, ORIGINAL, SCATTER);

    Timer tm;
    tm.start();

    // Mark all the vertices as not visited
    for(int i = 0; i < vertices_count; i++)
        _levels[i] = UNVISITED_VERTEX;

    // Create a queue for BFS
    list<int> queue;

    // Mark the current node as visited and enqueue it
    _levels[_source_vertex] = 1;
    queue.push_back(_source_vertex);

    while(!queue.empty())
    {
        // Dequeue a vertex from queue and print it
        int s = queue.front();
        queue.pop_front();

        int connections_count = _graph.get_outgoing_connections_count(s);
        for(int edge_pos = 0; edge_pos < connections_count; edge_pos++)
        {
            int v = _graph.get_outgoing_edge_dst(s, edge_pos);
            if (_levels[v] == UNVISITED_VERTEX)
            {
                _levels[v] = _levels[s] + 1;
                queue.push_back(v);
            }
        }
    }
    tm.end();


    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    performance_stats.print_algorithm_performance_stats("BFS (Top-Down, Sequential)", tm.get_time(), _graph.get_edges_count());
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
