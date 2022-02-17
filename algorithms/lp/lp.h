#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "gpu/lp_gpu.cuh"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define LP LabelPropagation
#define LP_DEFAULT_MAX_ITERATIONS 20

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// implements label propagation algorithm
class LabelPropagation
{
public:
    void allocate_result_memory(int _vertices_count, int **_labels);
    void free_result_memory    (int *_labels);

    void seq_lp(VectorCSRGraph &_graph, int *_labels, int _max_iterations = LP_DEFAULT_MAX_ITERATIONS);

    #ifdef __USE_GPU__
    void gpu_lp(VectorCSRGraph &_graph, int *_labels,
                GpuActiveConditionType _gpu_active_condition_type = ActivePassiveInner,
                int _max_iterations = LP_DEFAULT_MAX_ITERATIONS);
    #endif
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "lp.hpp"
#include "seq_lp.hpp"
#include "gpu_lp.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

