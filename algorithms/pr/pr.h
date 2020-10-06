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
private:
    double performance_per_iteration;
public:
    void allocate_result_memory  (int _vertices_count, float **_page_ranks);
    void free_result_memory      (float *_page_ranks);

    #ifdef __USE_NEC_SX_AURORA__
    void nec_page_rank(ExtendedCSRGraph &_graph,
                       float *_page_ranks,
                       float _convergence_factor = 1.0e-4,
                       int _max_iterations = 5);
    #endif

    #ifdef __USE_GPU__
    void gpu_page_rank(ExtendedCSRGraph &_graph,
                       float *_page_ranks,
                       float _convergence_factor = 1.0e-4,
                       int _max_iterations = 5,
                       AlgorithmTraversalType _traversal_direction = PULL_TRAVERSAL);
    #endif

    void seq_page_rank(ExtendedCSRGraph &_graph,
                       float *_page_ranks,
                       float _convergence_factor = 1.0e-4,
                       int _max_iterations = 5);

    double get_performance_per_iteration() {return performance_per_iteration;};
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "pr.hpp"
#include "seq_pr.hpp"
#include "nec_pr.hpp"
#include "gpu_pr.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
