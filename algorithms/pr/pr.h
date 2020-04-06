#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define PR PageRank<_TVertexValue, _TEdgeWeight>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
class PageRank
{
private:

public:
    static void allocate_result_memory  (int _vertices_count, float **_page_ranks);
    static void free_result_memory      (float *_page_ranks);

    #ifdef __USE_NEC_SX_AURORA__
    static void nec_page_rank(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                              float *_page_ranks,
                              float _convergence_factor = 1.0e-4,
                              int _max_iterations = 5);
    #endif

    static void seq_page_rank(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                              float *_page_ranks,
                              float _convergence_factor = 1.0e-4,
                              int _max_iterations = 5);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "pr.hpp"
#include "seq_pr.hpp"
#include "nec_pr.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
