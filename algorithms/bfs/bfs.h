#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <string>
#include "change_state/change_state.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

    #if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
    template <typename _T>
    static void nec_top_down(VectCSRGraph &_graph, VerticesArray<_T> &_levels, int _source_vertex);
    #endif

    template <typename _T>
    static void hardwired_do_bfs(VectCSRGraph &_graph, VerticesArray<_T> &_levels, int _source_vertex, BFS_GraphVE &_vector_extension,
                                 int *_buffer1, int *_buffer2, int alpha, int beta);

    template <typename _T>
    static void seq_top_down(VectCSRGraph &_graph, VerticesArray<_T> &_levels, int _source_vertex);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "seq_bfs.hpp"
#include "gpu_bfs.hpp"
#include "nec_bfs.hpp"
#include "change_state/change_state.hpp"
#include "hardwired_do_bfs.hpp"
#include "bfs_graph_ve.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

