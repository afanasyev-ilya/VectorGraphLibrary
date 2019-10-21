//
//  CSR_shard.h
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 09/08/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef shard_CSR_h
#define shard_CSR_h

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "shard_CSR_pointer_data.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TEdgeWeight>
class ShardCSR : public ShardBase<_TEdgeWeight>
{
private:
    long long *vertex_ptrs;
    int *dst_ids;
    _TEdgeWeight *weights;
    
    void resize(int _vertices_in_shard, long long _edges_in_shard);
public:
    ShardCSR();
    ~ShardCSR();
    
    void print();
    void init_shard_from_tmp_map();
    
    ShardCSRPointerData<_TEdgeWeight> get_pointers_data();
    
#ifdef __USE_GPU__
    void move_to_device();
    void move_to_host();
#endif
    
    void save_to_binary_file(FILE *_graph_file);
    void load_from_binary_file(FILE *_graph_file);
    
    void clear();
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "shard_CSR.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* CSR_shard_h */
