//
//  bf_edges_list.hpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 24/04/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef bf_edges_list_h
#define bf_edges_list_h

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ShortestPaths<_TVertexValue, _TEdgeWeight>::bellman_ford(EdgesListGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                              int _source_vertex, _TEdgeWeight *_distances)
{
    int vertices_count = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count();
    int *src_ids = _graph.get_src_ids();
    int *dst_ids = _graph.get_dst_ids();
    _TEdgeWeight *weights = _graph.get_weights();
    
    //double t1 = omp_get_wtime();
    double max_val = FLT_MAX;
    bool changes = true;
    int iterations_count = 0;
    for (int i = 0; i < vertices_count; i++)
    {
        _distances[i] = max_val;
    }
    _distances[_source_vertex] = 0;
        
    // do bellman-ford algorithm
    while (changes)
    {
        iterations_count++;
            
        changes = false;
            
        for (long long cur_edge = 0; cur_edge < edges_count; cur_edge++)
        {
            int src = src_ids[cur_edge];
            int dst = dst_ids[cur_edge];
            _TEdgeWeight weight = weights[cur_edge];
                
            _TEdgeWeight src_distance = _distances[src];
            _TEdgeWeight dst_distance = _distances[dst];
                
            if (dst_distance > src_distance + weight)
            {
                _distances[dst] = src_distance + weight;
                changes = true;
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* bf_edges_list_h */
