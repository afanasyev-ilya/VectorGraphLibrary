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

template <typename _TVertexValue, typename _TEdgeWeight>
void gpu_bellman_ford_wrapper(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, _TEdgeWeight *_distances,
                              int _source_vertex, int &_iterations_count);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* bellman_ford_gpu_h */
