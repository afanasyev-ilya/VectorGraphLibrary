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

#define SHARED_ELEMENTS_PER_THREAD 6

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class GraphPrimitivesGPU
{
private:
    cudaStream_t grid_processing_stream,block_processing_stream, warp_processing_stream, thread_processing_stream;
    cudaStream_t vwp_16_processing_stream, vwp_8_processing_stream, vwp_4_processing_stream, vwp_2_processing_stream;

    void split_frontier(FrontierGPU &_frontier);

    template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    void advance_sparse(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                        FrontierGPU &_frontier,
                        EdgeOperation edge_op,
                        VertexPreprocessOperation vertex_preprocess_op,
                        VertexPostprocessOperation vertex_postprocess_op);
public:
    GraphPrimitivesGPU();

    ~GraphPrimitivesGPU();

    template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    void advance(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                 FrontierGPU &_frontier,
                 EdgeOperation edge_op,
                 VertexPreprocessOperation vertex_preprocess_op,
                 VertexPostprocessOperation vertex_postprocess_op);

    template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation>
    void advance(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                 FrontierGPU &_frontier,
                 EdgeOperation edge_op);

    // creates new frontier, which satisfy user-defined "cond" condition
    template <typename _TVertexValue, typename _TEdgeWeight, typename Condition>
    void generate_new_frontier(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, FrontierGPU &_frontier, Condition &&cond);

    // performs user-defined "compute_op" operation for each element in the given frontier
    template <typename _TVertexValue, typename _TEdgeWeight, typename ComputeOperation>
    void compute(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, FrontierGPU &_frontier, ComputeOperation &&compute_op);

    // removes elements from current frontier, which satisfy user-defined "filter_cond" condition
    //template <typename _TVertexValue, typename _TEdgeWeight, typename FilterCondition>
    //void filter(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, FrontierGPU &_frontier, FilterCondition &&filter_cond);

    // performs reduction using user-defined "reduce_op" operation for each element in the given frontier
    template <typename _T, typename _TVertexValue, typename _TEdgeWeight, typename ReduceOperation>
    _T reduce(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, FrontierGPU &_frontier, ReduceOperation &&reduce_op, REDUCE_TYPE _reduce_type);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "advance.cu"
#include "advance_sparse.cu"
#include "compute.cu"
#include "filter.cu"
#include "reduce.cu"
#include "generate_new_frontier.cu"
#include "graph_primitives_gpu.cu"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////