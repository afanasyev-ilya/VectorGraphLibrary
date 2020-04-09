#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../../../common/gpu_API/cuda_error_handling.h"
#include "../../../architectures.h"
#include "../../../graph_representations/base_graph.h"
#include "../../../graph_representations/edges_list_graph/edges_list_graph.h"
#include "../../../graph_representations/extended_CSR_graph/extended_CSR_graph.h"
#include "graph_primitives_gpu.cuh"
#include <nvfunctional>

#include "../frontier/frontier_gpu.cuh"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class GraphPrimitivesGPU
{
private:
    cudaStream_t grid_processing_stream,block_processing_stream, warp_processing_stream, thread_processing_stream;

    void split_frontier(FrontierGPU &_frontier);
public:
    GraphPrimitivesGPU();

    ~GraphPrimitivesGPU();

    template <typename ComputeOperation>
    void compute(ComputeOperation compute_op, int _compute_size);

    template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    void advance(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                 FrontierGPU &_frontier,
                 EdgeOperation edge_op,
                 VertexPreprocessOperation vertex_preprocess_op,
                 VertexPostprocessOperation vertex_postprocess_op);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "traversal_kernels.cu"
#include "compute_kernels.cu"
#include "graph_primitives_gpu.cu"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////