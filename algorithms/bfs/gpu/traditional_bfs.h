//
//  traditional_bfs.h
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 12/06/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef traditional_bfs_h
#define traditional_bfs_h

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void gpu_direction_optimising_bfs_wrapper(long long *_first_part_ptrs,
                                          int *_first_part_sizes,
                                          int _vector_segments_count,
                                          long long *_vector_group_ptrs,
                                          int *_vector_group_sizes,
                                          int *_outgoing_ids,
                                          int _number_of_vertices_in_first_part,
                                          int *_levels,
                                          int _vertices_count,
                                          long long _edges_count,
                                          int _source_vertex);

void gpu_scan_test();

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* traditional_bfs_h */
