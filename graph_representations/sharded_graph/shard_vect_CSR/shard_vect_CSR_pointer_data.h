//
//  shard_vect_CSR_pointer_data.h
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 17/08/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef shard_vect_CSR_pointer_data_h
#define shard_vect_CSR_pointer_data_h

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TEdgeWeight>
struct ShardVectCSRPointerData
{
    int vertices_in_shard;
    long long *vector_group_ptrs;
    int *vector_group_sizes;
    int *dst_ids;
    _TEdgeWeight *weights;
    
    int *global_src_ids;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* shard_vect_CSR_pointer_data_h */
