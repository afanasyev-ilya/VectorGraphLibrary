//
//  shard_CSR_pointer_data.h
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 12/08/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef shard_CSR_pointer_data_h
#define shard_CSR_pointer_data_h

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TEdgeWeight>
struct ShardCSRPointerData
{
    int vertices_in_shard;
    int *global_src_ids;
    long long *vertex_ptrs;
    int *dst_ids;
    _TEdgeWeight *weights;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* shard_CSR_pointer_data_h */
