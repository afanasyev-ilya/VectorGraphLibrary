#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class GraphAbstractionsGPU : public GraphAbstractions
{
private:
    cudaStream_t grid_processing_stream,block_processing_stream, warp_processing_stream, thread_processing_stream;
    cudaStream_t vwp_16_processing_stream, vwp_8_processing_stream, vwp_4_processing_stream, vwp_2_processing_stream;

    template <typename ComputeOperation>
    void compute_worker(UndirectedCSRGraph &_graph, FrontierGPU &_frontier, ComputeOperation &&compute_op);

    //void split_frontier(FrontierGPU &_frontier);

    template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
    void GraphAbstractionsGPU::advance_worker(UndirectedCSRGraph &_graph,
                                              FrontierGPU &_frontier,
                                              EdgeOperation edge_op,
                                              VertexPreprocessOperation vertex_preprocess_op,
                                              VertexPostprocessOperation vertex_postprocess_op,
                                              bool _generate_frontier);


    /*int estimate_advance_work(UndirectedCSRGraph &_graph,
                              FrontierGPU &_frontier);*/
public:
    // attaches graph-processing API to the specific graph
    GraphAbstractionsGPU(VectCSRGraph &_graph, TraversalDirection _initial_traversal = SCATTER);
    GraphAbstractionsGPU(ShardedCSRGraph &_graph, TraversalDirection _initial_traversal = SCATTER);
    ~GraphAbstractionsGPU();

    // performs user-defined "edge_op" operation over all OUTGOING edges, neighbouring specified frontier
    template <typename EdgeOperation, typename VertexPreprocessOperation, typename VertexPostprocessOperation,
            typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
            typename CollectiveVertexPostprocessOperation>
    void scatter(VectCSRGraph &_graph,
                 FrontierGPU &_frontier,
                 EdgeOperation &&edge_op,
                 VertexPreprocessOperation &&vertex_preprocess_op,
                 VertexPostprocessOperation &&vertex_postprocess_op,
                 CollectiveEdgeOperation &&collective_edge_op,
                 CollectiveVertexPreprocessOperation &&collective_vertex_preprocess_op,
                 CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op);

    // performs user-defined "edge_op" operation over all OUTGOING edges, neighbouring specified frontier
    template <typename EdgeOperation>
    void scatter(VectCSRGraph &_graph, FrontierGPU &_frontier, EdgeOperation &&edge_op);

    // performs user-defined "compute_op" operation for each element in the given frontier
    template <typename ComputeOperation>
    void compute(VectCSRGraph &_graph,
                 FrontierGPU &_frontier,
                 ComputeOperation &&compute_op);

    // creates new frontier, which satisfy user-defined "cond" condition
    /*template <typename _TVertexValue, typename _TEdgeWeight, typename Condition>
    void generate_new_frontier(UndirectedCSRGraph &_graph, FrontierGPU &_frontier, Condition &&cond);

    // performs user-defined "compute_op" operation for each element in the given frontier
    template <typename _TVertexValue, typename _TEdgeWeight, typename ComputeOperation>
    void compute(UndirectedCSRGraph &_graph, FrontierGPU &_frontier, ComputeOperation &&compute_op);

    // performs reduction using user-defined "reduce_op" operation for each element in the given frontier
    template <typename _T, typename _TVertexValue, typename _TEdgeWeight, typename ReduceOperation>
    _T reduce(UndirectedCSRGraph &_graph, FrontierGPU &_frontier, ReduceOperation &&reduce_op, REDUCE_TYPE _reduce_type);*/
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __CUDA_INCLUDE__
#include "scatter.cu"
#include "advance.cu"
#include "compute.cu"
//#include "reduce.cu"
//#include "generate_new_frontier.cu"
#include "graph_abstractions_gpu.cu"
//#include "helpers.cu"
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////