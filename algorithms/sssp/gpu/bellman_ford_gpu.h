//
//  bf_gpu.h
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 01/05/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef bellman_ford_gpu_h
#define bellman_ford_gpu_h

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void gpu_bellman_ford_wrapper(long long *_first_part_ptrs, int *_first_part_sizes, int _vector_segments_count,
                              int _number_of_vertices_in_first_part,
                              long long *_vector_group_ptrs, int *_vector_group_sizes, int *outgoing_ids,
                              _T *_outgoing_weights, _T *device_distances, int _vertices_count, long long _edges_count,
                              int _source_vertex, int &_iterations_count);

template <typename _T>
void gpu_sharded_bellman_ford_wrapper(int _number_of_shards, void *_shards_data, _T *_distances,
                                      int _vertices_count, long long _edges_count, int &_iterations_count,
                                      ShardType _shard_type);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* bellman_ford_gpu_h */
