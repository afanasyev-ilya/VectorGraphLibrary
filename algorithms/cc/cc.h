#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define COMPONENT_UNSET -1
#define FIRST_COMPONENT 1
#define SINGLE_VERTEX_COMPONENT -2
#define DUO_VERTEX_COMPONENT -3
#define COMPONENT_FIRST_BFS_LEVEL -4

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define CC ConnectedComponents

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class ConnectedComponents
{
private:
    #if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
    template <typename _T>
    static int select_pivot(VGL_Graph &_graph, VGL_GRAPH_ABSTRACTIONS &_graph_API, VGL_FRONTIER &_frontier,
                            VerticesArray<_T> &_components);
    #endif
public:
    #if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
    template <typename _T>
    static void vgl_shiloach_vishkin(VGL_Graph &_graph, VerticesArray<_T> &_components);
    #endif

    #if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
    template <typename _T>
    static void vgl_bfs_based(VGL_Graph &_graph, VerticesArray<_T> &_components);
    #endif

    #ifdef __USE_GPU__
    template <typename _T>
    static void gpu_shiloach_vishkin(VGL_Graph &_graph, VerticesArray<_T> &_components);
    #endif

    template <typename _T>
    static void seq_bfs_based(VGL_Graph &_graph, VerticesArray<_T> &_components);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "shiloach_vishkin.hpp"
#include "bfs_based.hpp"
#include "seq_bfs_based.hpp"
#include "gpu_shiloach_vishkin.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
