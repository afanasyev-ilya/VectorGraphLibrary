#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define SSWP WidestPaths

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define vect_min(a,b) ((a)<(b)?(a):(b))
#define vect_max(a,b) ((a)>(b)?(a):(b))

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class WidestPaths
{
public:
    // architecture-independent API example. same function for NEC, multicore, GPU
    template <typename _T>
    static void vgl_dijkstra(VectCSRGraph &_graph, EdgesArray_Vect<_T> &_edges_capacities,
                             VerticesArray<_T> &_widths, int _source_vertex);

    template <typename _T>
    static void seq_dijkstra(VectCSRGraph &_graph, EdgesArray_Vect<_T> &_edges_capacities,
                             VerticesArray<_T> &_widths, int _source_vertex);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "nec_widest_paths.hpp"
#include "seq_widest_paths.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
