//
//  shard_base.h
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 09/08/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef shard_base_h
#define shard_base_h

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TEdgeWeight>
struct TmpMapEdge
{
    int dst_id;
    _TEdgeWeight weight;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TEdgeWeight>
class ShardBase
{
protected:
    int vertices_in_shard;
    long long edges_in_shard;
    
    int *global_src_ids;
    
    map< int,vector< TmpMapEdge<_TEdgeWeight> > > tmp_map_data;
public:
    ShardBase() {};
    ~ShardBase() {};
    
    void print_stats(int _total_vertices, long long _total_edges);
    void add_edge_to_tmp_map(int _src_id, int _dst_id, _TEdgeWeight _weights);
    
    int get_vertices_in_shard() { return vertices_in_shard; };
    long long get_edges_in_shard() { return edges_in_shard; };
    int *get_global_src_ids() { return global_src_ids; };
    
    virtual void print                   () = 0;
    virtual void init_shard_from_tmp_map () = 0;
    
    template<typename _T>
    void scatter_local_shard_data(_T *_local_data, _T *_global_data);
    
    template<typename _T>
    void gather_local_shard_data (_T *_local_data, _T *_global_data);
    
#ifdef __USE_GPU__
    virtual void move_to_device() = 0;
    virtual void move_to_host() = 0;
#endif
    
    virtual void save_to_binary_file(FILE *_graph_file) = 0;
    virtual void load_from_binary_file(FILE *_graph_file) = 0;
    
    virtual void clear() = 0;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "shard_base.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* shard_base_h */
