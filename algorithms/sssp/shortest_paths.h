//
//  shortest_paths.h
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 18/04/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef shortest_paths_h
#define shortest_paths_h

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_INTEL__
#include <limits>
#include <cfloat>
#endif

#ifdef __USE_NEC_SX_AURORA__
#include <ftrace.h>
#endif

#ifdef __USE_GPU__
#include "gpu/bellman_ford_gpu.h"
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
class ShortestPaths
{
private:
    static double process_csr_shard(ShardedGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                    _TEdgeWeight *_distances, _TEdgeWeight *_local_distances,
                                    int _shard_pos, int &_changes);
    static double process_vect_csr_shard(ShardedGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                         _TEdgeWeight *_distances, _TEdgeWeight *_local_distances,
                                         int _shard_pos, int &_changes);
    
    static void bellman_ford_kernel(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int _source_vertex,
                                    _TEdgeWeight *_distances, int &_changes, int &_iterations_count);
    
    static void print_performance_stats(long long _edges_count, int _iterations_count, double _total_time,
                                        double _gather_time, double _first_part_time,
                                        double _last_part_time, int _bytes_per_edge);
public:
    static void allocate_result_memory(int _vertices_count, _TEdgeWeight **_distances);
    static void free_result_memory    (_TEdgeWeight *_distances);
    
    static void reorder_result(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, _TEdgeWeight *_distances);
    static void reorder_result(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, _TEdgeWeight *_distances);

    static void bellman_ford(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_reversed_graph, int _source_vertex,
                             _TEdgeWeight *_distances);
    
    static void bellman_ford(EdgesListGraph<_TVertexValue, _TEdgeWeight> &_reversed_graph, int _source_vertex,
                             _TEdgeWeight *_distances);
    
    static void bellman_ford(ShardedGraph<_TVertexValue, _TEdgeWeight> &_reversed_graph, int _source_vertex,
                             _TEdgeWeight *_distances);
    
    static void bellman_ford(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_reversed_graph, int _source_vertex,
                             _TEdgeWeight *_distances);
    
    #ifdef __USE_GPU__
    static void gpu_bellman_ford(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_reversed_graph, int _source_vertex,
                                 _TEdgeWeight *_distances);
    static void gpu_bellman_ford(ShardedGraph<_TVertexValue, _TEdgeWeight> &_reversed_graph, int _source_vertex,
                                 _TEdgeWeight *_distances);
    #endif
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "shortest_paths.hpp"
#include "bf_vectorised_CSR.hpp"
#include "bf_edges_list.hpp"
#include "bf_sharded_CSR.hpp"
#include "bf_ext_CSR.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* shortest_paths_h */
