#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../../../common/gpu_API/cuda_error_handling.h"
#include "../../../architectures.h"
#include "../../../graph_representations/base_graph.h"
#include "../../../graph_representations/edges_list_graph/edges_list_graph.h"
#include "../../../graph_representations/extended_CSR_graph/extended_CSR_graph.h"
#include <nvfunctional>
#include "../frontier/frontier_gpu.cuh"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class GraphPrimitivesGPU
{
private:
    cudaStream_t grid_processing_stream,block_processing_stream, warp_processing_stream, thread_processing_stream;
    cudaStream_t vwp_16_processing_stream, vwp_8_processing_stream, vwp_4_processing_stream, vwp_2_processing_stream;

    void split_frontier(FrontierGPU &_frontier);

    template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    void advance_sparse(ExtendedCSRGraph &_graph,
                        FrontierGPU &_frontier,
                        EdgeOperation edge_op,
                        VertexPreprocessOperation vertex_preprocess_op,
                        VertexPostprocessOperation vertex_postprocess_op,
                        bool _generate_frontier = false);


    int estimate_advance_work(ExtendedCSRGraph &_graph,
                              FrontierGPU &_frontier);
public:
    GraphPrimitivesGPU();

    ~GraphPrimitivesGPU();

    template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    void advance(ExtendedCSRGraph &_graph,
                 FrontierGPU &_frontier,
                 EdgeOperation edge_op,
                 VertexPreprocessOperation vertex_preprocess_op,
                 VertexPostprocessOperation vertex_postprocess_op);

    template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation>
    void advance(ExtendedCSRGraph &_graph,
                 FrontierGPU &_frontier,
                 EdgeOperation edge_op);

    template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation, typename Condition>
    void advance(ExtendedCSRGraph &_graph,
                 FrontierGPU &_in_frontier,
                 EdgeOperation edge_op,
                 VertexPreprocessOperation vertex_preprocess_op,
                 VertexPostprocessOperation vertex_postprocess_op,
                 FrontierGPU &_out_frontier,
                 Condition &&cond);

    // creates new frontier, which satisfy user-defined "cond" condition
    template <typename _TVertexValue, typename _TEdgeWeight, typename Condition>
    void generate_new_frontier(ExtendedCSRGraph &_graph, FrontierGPU &_frontier, Condition &&cond);

    // performs user-defined "compute_op" operation for each element in the given frontier
    template <typename _TVertexValue, typename _TEdgeWeight, typename ComputeOperation>
    void compute(ExtendedCSRGraph &_graph, FrontierGPU &_frontier, ComputeOperation &&compute_op);

    // performs reduction using user-defined "reduce_op" operation for each element in the given frontier
    template <typename _T, typename _TVertexValue, typename _TEdgeWeight, typename ReduceOperation>
    _T reduce(ExtendedCSRGraph &_graph, FrontierGPU &_frontier, ReduceOperation &&reduce_op, REDUCE_TYPE _reduce_type);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "advance.cu"
#include "advance_sparse.cu"
#include "compute.cu"
#include "reduce.cu"
#include "generate_new_frontier.cu"
#include "graph_primitives_gpu.cu"
#include "helpers.cu"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////