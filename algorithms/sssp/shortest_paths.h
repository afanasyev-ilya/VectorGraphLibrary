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

enum AlgorithmFrontierType {
    ALL_ACTIVE = 1,
    PARTIAL_ACTIVE = 0
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define SSSP ShortestPaths<_TVertexValue, _TEdgeWeight>

template <typename _TVertexValue, typename _TEdgeWeight>
class ShortestPaths
{
private:
    #ifdef __USE_NEC_SX_AURORA__
    GraphPrimitivesNEC graph_API;
    FrontierNEC frontier;
    #endif

    void bellman_ford_kernel(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int _source_vertex,
                             _TEdgeWeight *_distances, int &_changes, int &_iterations_count);

    #ifdef __USE_NEC_SX_AURORA__
    void nec_dijkstra_all_active(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                 int _source_vertex, _TEdgeWeight *_distances);
    #endif

    #ifdef __USE_NEC_SX_AURORA__
    void nec_dijkstra_partial_active(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                     int _source_vertex, _TEdgeWeight *_distances);
    #endif
public:
    ShortestPaths(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph);

    void allocate_result_memory(int _vertices_count, _TEdgeWeight **_distances);
    void free_result_memory    (_TEdgeWeight *_distances);

    void reorder_result(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, _TEdgeWeight *_distances);

    void seq_dijkstra(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int _source_vertex,
                      _TEdgeWeight *_distances);

    #ifdef __USE_NEC_SX_AURORA__
    void nec_dijkstra(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int _source_vertex,
                      _TEdgeWeight *_distances, AlgorithmFrontierType _frontier_type = PARTIAL_ACTIVE);
    #endif

    #ifdef __USE_GPU__
    void gpu_dijkstra(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int _source_vertex,
                      _TEdgeWeight *_distances);
    #endif

};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "shortest_paths.hpp"
#include "gpu_shortest_paths.hpp"
#include "seq_shortest_paths.hpp"
#include "nec_shortest_paths.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

