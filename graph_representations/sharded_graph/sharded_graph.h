//
//  sharded_graph.h
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 09/08/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef sharded_graph_h
#define sharded_graph_h

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <map>
#include <iterator>
#include <vector>

#include "shard_base/shard_base.h"
#include "shard_CSR/shard_CSR.h"
#include "shard_vect_CSR/shard_vect_CSR.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace std;

enum ShardType {
    SHARD_CSR_TYPE,
    SHARD_VECT_CSR_TYPE
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


class ShardedGraph : public BaseGraph
{
private:
    int cache_size;
    int max_cached_vertices;
    int number_of_shards;
    ShardBase<_TEdgeWeight> **shards_data;
    
    ShardType type_of_shard;
    
    int get_shard_id(int _dst_id) { return _dst_id / max_cached_vertices; };
    
public:
    ShardedGraph(ShardType _type_of_shard, int _cache_size);
    ~ShardedGraph();
    
    void print();
    void print_stats();
    
    void resize(int _vertices_count, long long _edges_count, int _vertices_in_first_part = 0) {throw "not implemented yet";};
    void clear();
    
    void save_to_graphviz_file(string _file_name, VisualisationMode _visualisation_mode = VISUALISE_AS_DIRECTED) {throw "not implemented yet";};
    bool save_to_binary_file(string file_name);
    bool load_from_binary_file(string file_name);
    
    void import_graph(EdgesListGraph &_old_graph,
                      AlgorithmTraversalType _traversal_type = PULL_TRAVERSAL);
    
    int get_number_of_shards() { return number_of_shards; };
    ShardBase<_TEdgeWeight> **get_shards_data() { return shards_data; };
    
    template<typename _T>
    _T *allocate_local_shard_data();
    
    ShardType get_shard_type() { return type_of_shard; };
    
#ifdef __USE_GPU__
    void move_to_device();
    void move_to_host();
#endif
    
    void set_threads_count(int _threads_count) {};
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "sharded_graph.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* sharded_graph_h */
