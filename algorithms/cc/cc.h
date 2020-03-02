#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_INTEL__
#include <limits>
#include <cfloat>
#endif

#define COMPONENT_UNSET -1
#define FIRST_COMPONENT 1
#define SINGLE_VERTEX_COMPONENT -2
#define DUO_VERTEX_COMPONENT -3

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
class ConnectedComponents
{
private:
    GraphPrimitivesNEC graph_API;
    FrontierNEC frontier;
    FrontierNEC bfs_frontier;

    void performance_stats(string _name, double _time, long long _edges_count, int _iterations_count);
    void component_stats(int *_components, int _vertices_count);
public:
    ConnectedComponents(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph): frontier(_graph.get_vertices_count()), bfs_frontier(_graph.get_vertices_count()){};
    ~ConnectedComponents() {};

    void allocate_result_memory(int _vertices_count, int **_components);
    void free_result_memory    (int *_components);

    #ifdef __USE_NEC_SX_AURORA__
    void nec_shiloach_vishkin(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int *_components);
    #endif

    #ifdef __USE_NEC_SX_AURORA__
    void nec_bfs_based(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int *_components);
    #endif

    void seq_bfs_based(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int *_components);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "cc.hpp"
#include "nec_shiloach_vishkin.hpp"
#include "nec_bfs_based.hpp"
#include "seq_bfs_based.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
