#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../../common_datastructures/gpu_API/cuda_error_handling.h"
#include "../../architectures.h"
#include "../../graph_representations/base_graph.h"
#include "../../common_datastructures/gpu_API/gpu_arrays.h"
#include "../../graph_representations/edges_list_graph/edges_list_graph.h"
#include "../../graph_representations/vectorised_CSR_graph/vectorised_CSR_graph.h"
#include "../../graph_processing_API/gpu/graph_primitives_gpu.cuh"
#include <nvfunctional>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class GraphPrimitivesGPU
{
private:
    cudaStream_t grid_processing_stream,block_processing_stream, warp_processing_stream, thread_processing_stream;
public:
    GraphPrimitivesGPU();

    ~GraphPrimitivesGPU();

    template <typename InitOperation>
    void init(int size, InitOperation init_op);

    template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    void advance(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                 EdgeOperation edge_op,
                 VertexPreprocessOperation vertex_preprocess_op,
                 VertexPostprocessOperation vertex_postprocess_op);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_primitives_gpu.cu"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////