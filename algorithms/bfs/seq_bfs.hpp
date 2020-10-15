#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <queue>
#include <list>

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void BFS::seq_top_down(VectCSRGraph &_graph,
                       VerticesArray<int> &_levels,
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

        const long long edge_start = vertex_pointers[s];
        const int connections_count = vertex_pointers[s + 1] - vertex_pointers[s];

        for(int edge_pos = 0; edge_pos < connections_count; edge_pos++)
        {
            long long int global_edge_pos = edge_start + edge_pos;
            int v = adjacent_ids[global_edge_pos];
            if (_levels[v] == UNVISITED_VERTEX)
            {
                _levels[v] = _levels[s] + 1;
                queue.push_back(v);
            }
        }
    }

    tm.end();
    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    PerformanceStats::print_algorithm_performance_stats("BFS (Sequential Top-Down)", tm.get_time(), _graph.get_edges_count());
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
