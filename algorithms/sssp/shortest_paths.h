#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_MULTICORE__
#include <limits>
#include <cfloat>
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define SSSP ShortestPaths

#define vect_min(a,b) ((a)<(b)?(a):(b))

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// implements shortest paths algorithms
class ShortestPaths
{
public:
    // worker functions
    #if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
    template <typename _T>
    static double vgl_dijkstra_all_active_push(VGL_Graph &_graph,
                                               EdgesArray<_T> &_weights,
                                               VerticesArray<_T> &_distances,
                                               int _source_vertex);
    #endif

    #if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
    template <typename _T>
    static double vgl_dijkstra_all_active_pull(VGL_Graph &_graph,
                                               EdgesArray<_T> &_weights,
                                               VerticesArray<_T> &_distances,
                                               int _source_vertex);
    #endif

    #if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
    template <typename _T>
    static double vgl_dijkstra_partial_active(VGL_Graph &_graph,
                                              EdgesArray<_T> &_weights,
                                              VerticesArray<_T> &_distances,
                                              int _source_vertex);
    #endif

    #ifdef __USE_GPU__
    template <typename _T>
    static double gpu_dijkstra_partial_active(VGL_Graph &_graph,
                                            EdgesArray<_T> &_weights,
                                            VerticesArray<_T> &_distances,
                                            int _source_vertex);
    #endif

    #ifdef __USE_GPU__
    template <typename _T>
    static double gpu_dijkstra_all_active_push(VGL_Graph &_graph,
                                             EdgesArray<_T> &_weights,
                                             VerticesArray<_T> &_distances,
                                             int _source_vertex);
    #endif

    #ifdef __USE_GPU__
    template <typename _T>
    static double gpu_dijkstra_all_active_pull(VGL_Graph &_graph,
                                             EdgesArray<_T> &_weights,
                                             VerticesArray<_T> &_distances,
                                             int _source_vertex);
    #endif

    // --------------------------------- main interfaces ----------------------------------------
    template <typename _T>
    static double seq_dijkstra(VGL_Graph &_graph, EdgesArray<_T> &_weights, VerticesArray<_T> &_distances,
                               int _source_vertex);

    template <typename _T>
    static double vgl_dijkstra(VGL_Graph &_graph, EdgesArray<_T> &_weights, VerticesArray<_T> &_distances,
                               int _source_vertex, AlgorithmFrontierType _frontier_type = ALL_ACTIVE,
                               AlgorithmTraversalType _traversal_direction = PUSH_TRAVERSAL);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "gpu_shortest_paths.hpp"
#include "seq_shortest_paths.hpp"
#include "shortest_paths.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

