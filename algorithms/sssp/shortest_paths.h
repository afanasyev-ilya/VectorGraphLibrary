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

#define SSSP ShortestPaths<_TVertexValue, _TEdgeWeight>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
class ShortestPaths
{
private:
    GraphPrimitivesNEC graph_API;
    FrontierNEC frontier;

    #ifdef __USE_NEC_SX_AURORA__
    void nec_dijkstra_all_active(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                 _TEdgeWeight *_distances, int _source_vertex,
                                 TraversalDirection _traversal_direction);
    #endif

    #ifdef __USE_NEC_SX_AURORA__
    void nec_dijkstra_partial_active(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                     _TEdgeWeight *_distances,
                                     int _source_vertex);
    #endif

    void performance_stats(string _name, double _time, long long _edges_count, int _iterations_count);
public:
    ShortestPaths(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph);

    void allocate_result_memory(int _vertices_count, _TEdgeWeight **_distances);
    void free_result_memory    (_TEdgeWeight *_distances);

    void reorder_result(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, _TEdgeWeight *_distances);

    void seq_dijkstra(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, _TEdgeWeight *_distances,
                      int _source_vertex);

    #ifdef __USE_NEC_SX_AURORA__
    void nec_dijkstra(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, _TEdgeWeight *_distances,
                      int _source_vertex, AlgorithmFrontierType _frontier_type = PARTIAL_ACTIVE,
                      TraversalDirection _traversal_direction = PUSH_TRAVERSAL);
    #endif

    #ifdef __USE_GPU__
    void gpu_dijkstra(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                      _TEdgeWeight *_distances, int _source_vertex);
    #endif

    void nec_dijkstra_man(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, _TEdgeWeight *_distances, int _source_vertex);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "shortest_paths.hpp"
#include "gpu_shortest_paths.hpp"
#include "seq_shortest_paths.hpp"
#include "nec_shortest_paths.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

