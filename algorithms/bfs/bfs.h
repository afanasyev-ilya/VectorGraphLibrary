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

    BFS_GraphVE(VGL_Graph &_graph);
    ~BFS_GraphVE();
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class TransitiveClosure;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class BFS
{
private:
    template <typename _T>
    static void fast_vgl_top_down(VGL_Graph &_graph,
                                  VerticesArray<_T> &_levels,
                                  int _source_vertex,
                                  VGL_GRAPH_ABSTRACTIONS &_graph_API,
                                  VGL_FRONTIER &_frontier);
public:
    #ifdef __USE_GPU__
    template <typename _T>
    static double vgl_top_down(VGL_Graph &_graph, VerticesArray<_T> &_levels, int _source_vertex);
    #endif

    #if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
    template <typename _T>
    static double vgl_top_down(VGL_Graph &_graph, VerticesArray<_T> &_levels, int _source_vertex);
    #endif

    template <typename _T>
    static double seq_top_down(VGL_Graph &_graph, VerticesArray<_T> &_levels, int _source_vertex);

    friend class TransitiveClosure;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "seq_bfs.hpp"
#include "gpu_bfs.hpp"
#include "bfs.hpp"
#include "change_state/change_state.hpp"
#if defined(__USE_NEC_SX_AURORA__)
//#include "hardwired_do_bfs.hpp"
#endif
//#include "bfs_graph_ve.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

