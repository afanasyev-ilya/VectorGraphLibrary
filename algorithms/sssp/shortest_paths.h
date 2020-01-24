#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_INTEL__
#include <limits>
#include <cfloat>
#endif

#ifdef __USE_NEC_SX_AURORA__
#include <ftrace.h>
#endif

#ifdef __USE_GPU__
#include "gpu/dijkstra_gpu.cuh"
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
    
    static void print_performance_stats(long long _edges_count, int _iterations_count, double _wall_time);
public:
    static void allocate_result_memory(int _vertices_count, _TEdgeWeight **_distances);
    static void free_result_memory    (_TEdgeWeight *_distances);
    
    static void reorder_result(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, _TEdgeWeight *_distances);
    static void reorder_result(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, _TEdgeWeight *_distances);

    static void nec_dijkstra(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_reversed_graph, int _source_vertex,
                             _TEdgeWeight *_distances);
    
    static void seq_dijkstra(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_reversed_graph, int _source_vertex,
                             _TEdgeWeight *_distances);


    #ifdef __USE_NEC_SX_AURORA__
    static void lib_dijkstra(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_reversed_graph, int _source_vertex,
                             _TEdgeWeight *_distances);
    #endif

    #ifdef __USE_GPU__
    static void gpu_dijkstra(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int _source_vertex,
                             _TEdgeWeight *_distances);
    #endif
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "shortest_paths.hpp"
#include "gpu_shortest_paths.hpp"
#include "seq_shortest_paths.hpp"
#include "nec_shortest_paths.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
