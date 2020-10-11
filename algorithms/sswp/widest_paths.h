#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define SSWP WidestPaths

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define vect_min(a,b) ((a)<(b)?(a):(b))
#define vect_max(a,b) ((a)>(b)?(a):(b))

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


class WidestPaths
{
private:
    #ifdef __USE_NEC_SX_AURORA__
    GraphPrimitivesNEC graph_API;
    FrontierNEC frontier;
    _TEdgeWeight *class_old_widths;
    #endif

public:
    WidestPaths(UndirectedCSRGraph &_graph);
    ~WidestPaths();

    void allocate_result_memory(int _vertices_count, _TEdgeWeight **_widths);
    void free_result_memory    (_TEdgeWeight *_widths);

    #ifdef __USE_NEC_SX_AURORA__
    void nec_dijkstra(UndirectedCSRGraph &_graph, _TEdgeWeight *_widths, int _source_vertex,
                      AlgorithmTraversalType _traversal_direction);
    #endif

    void seq_dijkstra(UndirectedCSRGraph &_graph, _TEdgeWeight *_widths, int _source_vertex);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "widest_paths.hpp"
#include "nec_widest_paths.hpp"
#include "seq_widest_paths.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
