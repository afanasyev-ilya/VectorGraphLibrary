#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_INTEL__
#include <limits>
#include <cfloat>
#endif

#ifdef __USE_GPU__
#include "gpu/dijkstra_gpu.cuh"
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define SSSP ShortestPaths

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class ShortestPaths
{
private:
    #ifdef __USE_NEC_SX_AURORA__
    template <typename _T>
    static void nec_dijkstra_all_active_push(VectCSRGraph &_graph,
                                             EdgesArray<_T> &_weights,
                                             VerticesArray<_T> &_distances,
                                             int _source_vertex);
    #endif

    #ifdef __USE_NEC_SX_AURORA__
    template <typename _T>
    static void nec_dijkstra_all_active_pull(VectCSRGraph &_graph,
                                             EdgesArray<_T> &_weights,
                                             VerticesArray<_T> &_distances,
                                             int _source_vertex);
    #endif

    #ifdef __USE_NEC_SX_AURORA__
    template <typename _T>
    static void nec_dijkstra_partial_active(VectCSRGraph &_graph,
                                            EdgesArray<_T> &_weights,
                                            VerticesArray<_T> &_distances,
                                            int _source_vertex);
    #endif
public:
    template <typename _T>
    static void seq_dijkstra(VectCSRGraph &_graph, EdgesArray<_T> &_weights, VerticesArray<_T> &_distances,
                             int _source_vertex);

    #ifdef __USE_NEC_SX_AURORA__
    template <typename _T>
    static void nec_dijkstra(VectCSRGraph &_graph, EdgesArray<_T> &_weights, VerticesArray<_T> &_distances,
                             int _source_vertex, AlgorithmFrontierType _frontier_type = ALL_ACTIVE,
                             AlgorithmTraversalType _traversal_direction = PUSH_TRAVERSAL);
    #endif

    #ifdef __USE_NEC_SX_AURORA__
    template <typename _T>
    static void nec_dijkstra(ShardedCSRGraph &_graph,
                             EdgesArray<_T> &_weights,
                             VerticesArray<_T> &_distances,
                             int _source_vertex);
    #endif

    #ifdef __USE_GPU__
    template <typename _T>
    static void gpu_dijkstra(VectCSRGraph &_graph, EdgesArray<_T> &_weights, VerticesArray<_T> &_distances,
                             int _source_vertex, AlgorithmFrontierType _frontier_type = ALL_ACTIVE,
                             AlgorithmTraversalType _traversal_direction = PUSH_TRAVERSAL);
    #endif
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "gpu_shortest_paths.hpp"
#include "seq_shortest_paths.hpp"
#include "nec_shortest_paths.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

