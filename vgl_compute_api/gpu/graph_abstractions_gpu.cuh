#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__forceinline__ __device__ unsigned lane_id()
{
    unsigned ret;
    asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}

__forceinline__ __device__ unsigned warp_id()
{
    // this is not equal to threadIdx.x / 32
    unsigned ret;
    asm volatile ("mov.u32 %0, %warpid;" : "=r"(ret));
    return ret;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class GraphAbstractionsGPU : public GraphAbstractions
{
private:
    cudaStream_t grid_processing_stream,block_processing_stream, warp_processing_stream, thread_processing_stream;
    cudaStream_t vwp_16_processing_stream, vwp_8_processing_stream, vwp_4_processing_stream, vwp_2_processing_stream;

    template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
    void advance_worker(VectorCSRGraph &_graph,
                        VGL_Frontier &_frontier,
                        EdgeOperation edge_op,
                        VertexPreprocessOperation vertex_preprocess_op,
                        VertexPostprocessOperation vertex_postprocess_op,
                        bool _generate_frontier);

    template <typename EdgeOperation>
    void advance_worker(EdgesListGraph &_graph,
                        EdgeOperation &&edge_op);

    template <typename ComputeOperation>
    void compute_container_call(VGL_Graph &_graph,
                                VGL_Frontier &_frontier,
                                ComputeOperation &&compute_op);

    template <typename ComputeOperation, typename GraphContainer, typename FrontierContainer>
    void compute_worker(GraphContainer &_graph, FrontierContainer &_frontier, ComputeOperation &&compute_op);

    template <typename _T, typename ReduceOperation>
    _T reduce_worker(VectorCSRGraph &_graph,
                     VGL_Frontier &_frontier,
                     ReduceOperation &&reduce_op,
                     REDUCE_TYPE _reduce_type);
public:
    // attaches graph-processing API to the specific graph
    GraphAbstractionsGPU(VGL_Graph &_graph, TraversalDirection _initial_traversal = SCATTER);
    ~GraphAbstractionsGPU();

    // performs user-defined "edge_op" operation over all OUTGOING edges, neighbouring specified frontier
    template <typename EdgeOperation, typename VertexPreprocessOperation, typename VertexPostprocessOperation,
            typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
            typename CollectiveVertexPostprocessOperation>
    void scatter(VGL_Graph &_graph,
                 VGL_Frontier &_frontier,
                 EdgeOperation &&edge_op,
                 VertexPreprocessOperation &&vertex_preprocess_op,
                 VertexPostprocessOperation &&vertex_postprocess_op,
                 CollectiveEdgeOperation &&collective_edge_op,
                 CollectiveVertexPreprocessOperation &&collective_vertex_preprocess_op,
                 CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op);

    // performs user-defined "edge_op" operation over all OUTGOING edges, neighbouring specified frontier
    template <typename EdgeOperation>
    void scatter(VGL_Graph &_graph, VGL_Frontier &_frontier, EdgeOperation &&edge_op);

    // performs user-defined "edge_op" operation over all OUTGOING edges, neighbouring specified frontier
    template <typename EdgeOperation, typename VertexPreprocessOperation, typename VertexPostprocessOperation,
            typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
            typename CollectiveVertexPostprocessOperation>
    void gather(VGL_Graph &_graph,
                VGL_Frontier &_frontier,
                EdgeOperation &&edge_op,
                VertexPreprocessOperation &&vertex_preprocess_op,
                VertexPostprocessOperation &&vertex_postprocess_op,
                CollectiveEdgeOperation &&collective_edge_op,
                CollectiveVertexPreprocessOperation &&collective_vertex_preprocess_op,
                CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op);

    // performs user-defined "edge_op" operation over all OUTGOING edges, neighbouring specified frontier
    template <typename EdgeOperation>
    void gather(VGL_Graph &_graph, VGL_Frontier &_frontier, EdgeOperation &&edge_op);

    // performs user-defined "compute_op" operation for each element in the given frontier
    template <typename ComputeOperation>
    void compute(VGL_Graph &_graph, VGL_Frontier &_frontier, ComputeOperation &&compute_op);

    // creates new frontier, which satisfy user-defined "cond" condition
    template <typename Condition>
    void generate_new_frontier(VGL_Graph &_graph, VGL_Frontier &_frontier, Condition &&cond);

    // performs reduction using user-defined "reduce_op" operation for each element in the given frontier
    template <typename _T, typename ReduceOperation>
    _T reduce(VGL_Graph &_graph, VGL_Frontier &_frontier, ReduceOperation &&reduce_op, REDUCE_TYPE _reduce_type);

    friend class GraphAbstractions;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "scatter.cu"
#include "gather.cu"
#include "advance.cu"
#include "compute.cu"
//#include "reduce.cu"
#include "generate_new_frontier.cu"
#include "graph_abstractions_gpu.cu"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////