#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "gpu/lp_gpu.cuh"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define LP LabelPropagation<_TVertexValue, _TEdgeWeight>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
class LabelPropagation
{
public:
    void allocate_result_memory(int _vertices_count, int **_labels);
    void free_result_memory    (int *_labels);

    void seq_lp(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int *_labels);

    #ifdef __USE_GPU__
    void gpu_lp(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int *_labels);
    #endif
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "lp.hpp"
#include "seq_lp.hpp"
#include "gpu_lp.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

