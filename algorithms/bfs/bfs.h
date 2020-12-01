#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <string>
#include "change_state/change_state.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class BFS
{
private:
    #ifdef __USE_NEC_SX_AURORA__
    static void nec_top_down_compute_step(VectCSRGraph &_graph,
                                          VerticesArray<int> &_levels,
                                          int _current_level,
                                          int &_vis,
                                          int &_in_lvl,
                                          bool _compute_stats);
    #endif

    #ifdef __USE_NEC_SX_AURORA__
    static void nec_bottom_up_compute_step(VectCSRGraph &_graph,
                                           VerticesArray<int> &_levels,
                                           int *_connections_array,
                                           int _current_level,
                                           int &_vis,
                                           int &_in_lvl,
                                           bool _use_vector_extension);
    #endif
public:
    #ifdef __USE_GPU__
    template <typename _T>
    static void gpu_top_down(VectCSRGraph &_graph, VerticesArray<_T> &_levels, int _source_vertex);
    #endif

    #ifdef __USE_NEC_SX_AURORA__
    template <typename _T>
    static void nec_direction_optimizing(VectCSRGraph &_graph, VerticesArray<_T> &_levels, int _source_vertex);
    #endif

    #ifdef __USE_NEC_SX_AURORA__
    template <typename _T>
    static void nec_top_down(VectCSRGraph &_graph, VerticesArray<_T> &_levels, int _source_vertex);
    #endif

    template <typename _T>
    static void seq_top_down(VectCSRGraph &_graph, VerticesArray<_T> &_levels, int _source_vertex);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "seq_bfs.hpp"
#include "gpu_bfs.hpp"
#include "nec_bfs.hpp"
#include "change_state/change_state.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

