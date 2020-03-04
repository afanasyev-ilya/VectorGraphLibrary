//
//  shard_base.hpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 09/08/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef shard_base_hpp
#define shard_base_hpp

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TEdgeWeight>
void ShardBase<_TEdgeWeight>::add_edge_to_tmp_map(int _src_id, int _dst_id, _TEdgeWeight _weight)
{
    TmpMapEdge<_TEdgeWeight> tmp_edge;
    tmp_edge.dst_id = _dst_id;
    tmp_edge.weight = _weight;
    
    tmp_map_data[_src_id].push_back(tmp_edge);
    vertices_in_shard = tmp_map_data.size();
    edges_in_shard++;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double calc_percent(double _a, double _b)
{
    return 100.0*(_a/_b);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TEdgeWeight>
void ShardBase<_TEdgeWeight>::print_stats(int _total_vertices, long long _total_edges)
{
    cout << "vertices: " << vertices_in_shard << ", " << calc_percent(vertices_in_shard, _total_vertices) << "%" << endl;
    cout << "edges: " << edges_in_shard << ", " << calc_percent(edges_in_shard, _total_edges) << "%" << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TEdgeWeight>
template<typename _T>
void ShardBase<_TEdgeWeight>::gather_local_shard_data(_T *_local_data, _T *_global_data)
{
    #pragma omp parallel for schedule(static)
    #pragma simd
    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    for(int i = 0; i < vertices_in_shard; i++)
    {
        _local_data[i] = _global_data[global_src_ids[i]];
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TEdgeWeight>
template<typename _T>
void ShardBase<_TEdgeWeight>::scatter_local_shard_data(_T *_local_data, _T *_global_data)
{
    #pragma omp parallel for schedule(static)
    #pragma simd
    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    for(int i = 0; i < vertices_in_shard; i++)
    {
        _global_data[global_src_ids[i]] = _local_data[i];
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* shard_base_h */
