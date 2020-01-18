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

template <typename _TVertexValue, typename _TEdgeWeight>
void gpu_direction_optimising_bfs_wrapper(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int *_levels,
                                          int _source_vertex, int &_iterations_count, int *_active_ids, int *_active_vertices_buffer);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* traditional_bfs_h */
