#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <string>
#include "change_state/change_state.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define BFS_VE_SIZE 4
#define BOTTOM_UP_THRESHOLD 5

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct BFS_GraphVE
{
    int ve_vertices_count;
    int ve_edges_per_vertex;
    int *ve_dst_ids;

    BFS_GraphVE(VectCSRGraph &_graph);
    ~BFS_GraphVE();
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class BFS
{
public:
    #ifdef __USE_GPU__
    template <typename _T>
    static void gpu_top_down(VectCSRGraph &_graph, VerticesArray<_T> &_levels, int _source_vertex);
    #endif

    #ifdef __USE_NEC_SX_AURORA__
    template <typename _T>
    static void nec_direction_optimizing(VectCSRGraph &_graph, VerticesArray<_T> &_levels, int _source_vertex,
                                         BFS_GraphVE &_vector_extension, DirectionType _direction,
                                         int first_switch, int second_switch);
    #endif

    #ifdef __USE_NEC_SX_AURORA__
    template <typename _T>
    static void nec_top_down(VectCSRGraph &_graph, VerticesArray<_T> &_levels, int _source_vertex);
    #endif

    template <typename _T>
    static void manually_optimised_nec_bfs(VectCSRGraph &_graph, VerticesArray<int> &_levels, int _source_vertex, BFS_GraphVE &_vector_extension);

    template <typename _T>
    static void seq_top_down(VectCSRGraph &_graph, VerticesArray<_T> &_levels, int _source_vertex);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "seq_bfs.hpp"
#include "gpu_bfs.hpp"
#include "nec_bfs.hpp"
#include "change_state/change_state.hpp"
#include "manually_optimized.hpp"
#include "bfs_graph_ve.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

