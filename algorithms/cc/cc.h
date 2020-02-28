#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_INTEL__
#include <limits>
#include <cfloat>
#endif

#ifdef __USE_NEC_SX_AURORA__
#include <ftrace.h>
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
class ConnectedComponents
{
public:
    ConnectedComponents() {};
    ~ConnectedComponents() {};

    void allocate_result_memory(int _vertices_count, int **_components);
    void free_result_memory    (int *_components);

    #ifdef __USE_NEC_SX_AURORA__
    void nec_shiloach_vishkin(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int *_components);
    #endif

    void seq_bfs_based(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int *_components);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "cc.hpp"
#include "shiloach_vishkin.hpp"
#include "seq_bfs_based.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
