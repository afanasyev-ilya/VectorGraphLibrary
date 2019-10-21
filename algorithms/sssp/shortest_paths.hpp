//
//  shortest_paths.hpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 18/04/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef shortest_paths_hpp
#define shortest_paths_hpp

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ShortestPaths<_TVertexValue, _TEdgeWeight>::allocate_result_memory(int _vertices_count, _TEdgeWeight **_distances)
{
    *_distances = new _TEdgeWeight[_vertices_count];
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ShortestPaths<_TVertexValue, _TEdgeWeight>::free_result_memory(_TEdgeWeight *_distances)
{
    delete[] _distances;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ShortestPaths<_TVertexValue, _TEdgeWeight>::reorder_result(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                                _TEdgeWeight *_distances)
{
    int vertices_count = _graph.get_vertices_count();
    int *reordered_ids = _graph.get_reordered_vertex_ids();
    
    _TEdgeWeight *tmp_distances = new _TEdgeWeight[vertices_count];
    
    for(int i = 0; i < vertices_count; i++)
    {
        tmp_distances[i] = _distances[reordered_ids[i]];
    }
    
    for(int i = 0; i < vertices_count; i++)
    {
        _distances[i] = tmp_distances[i];
    }
    
    delete []tmp_distances;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ShortestPaths<_TVertexValue, _TEdgeWeight>::reorder_result(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                                _TEdgeWeight *_distances)
{
    int vertices_count = _graph.get_vertices_count();
    int *reordered_ids = _graph.get_reordered_vertex_ids();
    
    _TEdgeWeight *tmp_distances = new _TEdgeWeight[vertices_count];
    
    for(int i = 0; i < vertices_count; i++)
    {
        tmp_distances[i] = _distances[reordered_ids[i]];
    }
    
    for(int i = 0; i < vertices_count; i++)
    {
        _distances[i] = tmp_distances[i];
    }
    
    delete []tmp_distances;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ShortestPaths<_TVertexValue, _TEdgeWeight>::print_performance_stats(long long _edges_count,
                                                                         int _iterations_count,
                                                                         double _total_time,
                                                                         double _gather_time,
                                                                         double _first_part_time,
                                                                         double _last_part_time,
                                                                         int _bytes_per_edge)
{
    cout << "BF total time: " << _total_time * 1000.0 << " ms" << endl;
    cout << "BF Wall perf: " << ((double)_edges_count) / (_total_time * 1e6) << " MFLOPS" << endl;
    cout << "BF gather Perf: " << _iterations_count * ((double)_edges_count) / (_gather_time * 1e6) << " MFLOPS" << endl;
    cout << "BF process data Perf: " << _iterations_count * ((double)_edges_count) / ((_first_part_time + _last_part_time) * 1e6) << " MFLOPS" << endl;
    cout << "BF iterations count: " << _iterations_count << endl;
    cout << "BF Perf per iteration: " << _iterations_count * ((double)_edges_count) / (_total_time * 1e6) << " MFLOPS" << endl;
    
    int bytes_in_gather = sizeof(int) + 2.0*sizeof(_TEdgeWeight);
    cout << "BF bandwidth: " << ((double)_iterations_count)*((double)_edges_count * _bytes_per_edge) / (_total_time * 1e9) << " gb/s" << endl;
    cout << "BF gather bandwidth: " << ((double)_iterations_count)*((double)_edges_count * bytes_in_gather) / (_gather_time * 1e9) << " gb/s" << endl << endl << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* shortest_paths_hpp */
