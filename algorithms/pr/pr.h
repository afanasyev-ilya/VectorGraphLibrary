#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define PR PageRank

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// implements page rank algorithm
class PageRank
{
public:
    #if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
    template <typename _T>
    static double vgl_page_rank(VGL_Graph &_graph,
                                VerticesArray<_T> &_page_ranks,
                                _T _convergence_factor = 1.0e-4,
                                int _max_iterations = 5);
    #endif

    #ifdef __USE_GPU__
    template <typename _T>
    static double vgl_page_rank(VGL_Graph &_graph,
                                VerticesArray<_T> &_page_ranks,
                                _T _convergence_factor = 1.0e-4,
                                int _max_iterations = 5,
                                AlgorithmTraversalType _traversal_direction = PULL_TRAVERSAL);
    #endif

    template <typename _T>
    static double seq_page_rank(VGL_Graph &_graph,
                                VerticesArray<_T> &_page_ranks,
                                _T _convergence_factor = 1.0e-4,
                                int _max_iterations = 5);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "seq_pr.hpp"
#include "pr.hpp"
#include "gpu_pr.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
