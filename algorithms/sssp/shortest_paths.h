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
    double performance;

    #ifdef __USE_NEC_SX_AURORA__
    GraphPrimitivesNEC graph_API;
    FrontierNEC frontier;
    _TEdgeWeight *class_old_distances;
    #endif

    #ifdef __USE_NEC_SX_AURORA__
    void nec_dijkstra_all_active(ExtendedCSRGraph &_graph,
                                 _TEdgeWeight *_distances, int _source_vertex,
                                 AlgorithmTraversalType _traversal_direction);
    #endif

    #ifdef __USE_NEC_SX_AURORA__
    void nec_dijkstra_partial_active(ExtendedCSRGraph &_graph,
                                     _TEdgeWeight *_distances,
                                     int _source_vertex);
    #endif
public:
    ShortestPaths(ExtendedCSRGraph &_graph);
    ShortestPaths(EdgesListGraph &_graph);
    ~ShortestPaths();

    void allocate_result_memory(int _vertices_count, _TEdgeWeight **_distances);
    void free_result_memory    (_TEdgeWeight *_distances);

    void reorder_result(ExtendedCSRGraph &_graph, _TEdgeWeight *_distances);

    void seq_dijkstra(ExtendedCSRGraph &_graph, _TEdgeWeight *_distances,
                      int _source_vertex);

    void seq_dijkstra(VectCSRGraph &_graph,
                      EdgesArrayNec<_TVertexValue, _TEdgeWeight, _TEdgeWeight> &_weights,
                      _TEdgeWeight *_distances,
                      int _source_vertex);

    #ifdef __USE_NEC_SX_AURORA__
    void nec_dijkstra(ExtendedCSRGraph &_graph, _TEdgeWeight *_distances,
                      int _source_vertex, AlgorithmFrontierType _frontier_type = ALL_ACTIVE,
                      AlgorithmTraversalType _traversal_direction = PUSH_TRAVERSAL);
    #endif

    #ifdef __USE_NEC_SX_AURORA__
    void nec_dijkstra(VectCSRGraph &_graph,
                      EdgesArrayNec<_TVertexValue, _TEdgeWeight, _TEdgeWeight> &_weights,
                      _TEdgeWeight *_distances,
                      int _source_vertex);
    #endif

    #ifdef __USE_NEC_SX_AURORA__
    void nec_bellamn_ford(EdgesListGraph &_graph, _TEdgeWeight *_distances,
                          int _source_vertex);
    #endif

    #ifdef __USE_GPU__
    void gpu_dijkstra(ExtendedCSRGraph &_graph,
                      _TEdgeWeight *_distances, int _source_vertex,
                      AlgorithmFrontierType _frontier_type = PARTIAL_ACTIVE,
                      AlgorithmTraversalType _traversal_direction = PUSH_TRAVERSAL);
    #endif

    double get_performance() { return performance; };
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "shortest_paths.hpp"
#include "gpu_shortest_paths.hpp"
#include "seq_shortest_paths.hpp"
#include "nec_shortest_paths.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

