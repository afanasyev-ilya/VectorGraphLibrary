#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define COMPONENT_UNSET -1
#define FIRST_COMPONENT 1
#define SINGLE_VERTEX_COMPONENT -2
#define DUO_VERTEX_COMPONENT -3

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define CC ConnectedComponents

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class ConnectedComponents
{
public:
    #ifdef __USE_NEC_SX_AURORA__
    template <typename _T>
    static void nec_shiloach_vishkin(VectCSRGraph &_graph, VerticesArray<_T> &_components);
    #endif

    #ifdef __USE_NEC_SX_AURORA__
    //void nec_bfs_based(UndirectedCSRGraph &_graph, int *_components);
    #endif

    #ifdef __USE_GPU__
    template <typename _T>
    static void gpu_shiloach_vishkin(VectCSRGraph &_graph, VerticesArray<_T> &_components);
    #endif

    template <typename _T>
    static void seq_bfs_based(VectCSRGraph &_graph, VerticesArray<_T> &_components);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "nec_shiloach_vishkin.hpp"
//#include "nec_bfs_based.hpp"
#include "seq_bfs_based.hpp"
#include "gpu_shiloach_vishkin.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
