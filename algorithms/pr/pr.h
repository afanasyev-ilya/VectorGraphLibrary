#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define PR PageRank<_TVertexValue, _TEdgeWeight>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
class PageRank
{
private:
    void performance_stats(string _name, double _time, long long _edges_count, int _iterations_count);
public:
    void allocate_result_memory  (int _vertices_count, float **_page_ranks);
    void free_result_memory      (float *_page_ranks);

    #ifdef __USE_NEC_SX_AURORA__
    void nec_page_rank(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                       float *_page_ranks,
                       float _convergence_factor = 1.0e-4,
                       int _max_iterations = 5);
    #endif

    void seq_page_rank(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                       float *_page_ranks,
                       float _convergence_factor = 1.0e-4,
                       int _max_iterations = 5);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "pr.hpp"
#include "seq_pr.hpp"
#include "nec_pr.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
