//
//  cache_metric.h
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 24/05/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef cache_metrics_h
#define cache_metrics_h

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double analyse_dst_ptrs_for_cache_metric(vector<int> &_dst_ptrs_data, int _vertices_count,
                                         int _number_of_vertices_in_cache, long long _edges_count)
{
    long long number_of_edges_in_cache = 0;
    for(int i = 0; i < min(_number_of_vertices_in_cache, _vertices_count); i++)
        number_of_edges_in_cache += _dst_ptrs_data[i];
    
    cout << "cached vertices: " << _number_of_vertices_in_cache << endl;
    cout << "total  vertices: " << _vertices_count << endl;
    cout << "cached edges: " << number_of_edges_in_cache << endl;
    cout << "total  edges: " << _edges_count << endl;
    double cache_metric = ((double)number_of_edges_in_cache) / ((double)_edges_count);
    cout << "cache metric: " << cache_metric << endl << endl;
    
    return cache_metric;
}

template <typename _TVertexValue, typename _TEdgeWeight>
void calculate_cache_metric(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph)
{
    LOAD_VECTORISED_CSR_GRAPH_REVERSE_DATA(_graph)
    
    vector<int> dst_ptrs_data(_graph.get_vertices_count());
    collect_number_of_accesses_per_vertex(_graph, dst_ptrs_data);

    analyse_dst_ptrs_for_cache_metric(dst_ptrs_data, _graph.get_vertices_count(), CACHED_VERTICES,
                                      _graph.get_edges_count());
    
    int max_cached_vertices = 16 * 1e6 / sizeof(int);
    analyse_dst_ptrs_for_cache_metric(dst_ptrs_data, _graph.get_vertices_count(), max_cached_vertices,
                                      _graph.get_edges_count());
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* cache_metrics_h */
