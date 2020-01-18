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
                                                                         double _wall_time)
{
    int bytes_per_edge = sizeof(int) + 2*sizeof(_TEdgeWeight);
    cout << "Time               : " << _wall_time << endl;
    cout << "Performance        : " << ((double)_edges_count) / (_wall_time * 1e6) << " MFLOPS" << endl;
    cout << "Iteration    count : " << _iterations_count << endl;
    cout << "Perf. per iteration: " << _iterations_count * ((double)_edges_count) / (_wall_time * 1e6) << " MFLOPS" << endl;
    cout << "Bandwidth          : " << _iterations_count*((double)_edges_count * (bytes_per_edge)) / (_wall_time * 1e9) << " GB/s" << endl;
    cout << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* shortest_paths_hpp */
