#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
#include "gpu/pr_gpu.cuh"
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define PR PageRank

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class PageRank
{
public:
    #ifdef __USE_NEC_SX_AURORA__
    template <typename _T>
    static void nec_page_rank(VectCSRGraph &_graph,
                              VerticesArray<_T> &_page_ranks,
                              _T _convergence_factor = 1.0e-4,
                              int _max_iterations = 5);
    #endif

    #ifdef __USE_GPU__
    template <typename _T>
    static void gpu_page_rank(VectCSRGraph &_graph,
                              VerticesArray<_T> &_page_ranks,
                              _T _convergence_factor = 1.0e-4,
                              int _max_iterations = 5);
    #endif

    template <typename _T>
    static void seq_page_rank(VectCSRGraph &_graph,
                              VerticesArray<_T> &_page_ranks,
                              _T _convergence_factor = 1.0e-4,
                              int _max_iterations = 5);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "seq_pr.hpp"
#include "nec_pr.hpp"
#include "gpu_pr.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
