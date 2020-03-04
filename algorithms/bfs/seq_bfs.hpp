#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <queue>
#include <list>

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void BFS<_TVertexValue, _TEdgeWeight>::seq_top_down(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int *_levels, int _source_vertex)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

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

        const long long edge_start = outgoing_ptrs[s];
        const int connections_count = outgoing_ptrs[s + 1] - outgoing_ptrs[s];

        for(int edge_pos = 0; edge_pos < connections_count; edge_pos++)
        {
            long long int global_edge_pos = edge_start + edge_pos;
            int v = outgoing_ids[global_edge_pos];
            if (_levels[v] == UNVISITED_VERTEX)
            {
                _levels[v] = _levels[s] + 1;
                queue.push_back(v);
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
