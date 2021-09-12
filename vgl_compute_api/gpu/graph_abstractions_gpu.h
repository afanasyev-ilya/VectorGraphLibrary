#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/copy.h>
#include "helpers.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class GraphAbstractionsGPU : public GraphAbstractions
{
private:
    cudaStream_t stream_1, stream_2, stream_3, stream_4, stream_5, stream_6;
    double *reduce_buffer;

    // current the number of vertices, neighbouring a frontier (for Advance perf)
    long long count_frontier_neighbours(VGL_Graph &_graph, VGL_Frontier &_frontier);
    bool use_safe_stores;

    // compute inner implementation
    template <typename ComputeOperation, typename GraphContainer, typename FrontierContainer>
    void compute_worker(GraphContainer &_graph,
                        FrontierContainer &_frontier,
                        ComputeOperation &&compute_op);

    // reduce inner implementation
    template <typename _T, typename ReduceOperation, typename GraphContainer, typename FrontierContainer>
    void reduce_worker(GraphContainer &_graph,
                       FrontierContainer &_frontier,
                       ReduceOperation &&reduce_op,
                       REDUCE_TYPE _reduce_type,
                       _T &_result);

    // advance inner implementation
    template<typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation, typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
            typename CollectiveVertexPostprocessOperation>
    void advance_worker(EdgesListGraph &_graph,
                        FrontierEdgesList &_frontier,
                        EdgeOperation &&edge_op,
                        VertexPreprocessOperation &&vertex_preprocess_op,
                        VertexPostprocessOperation &&vertex_postprocess_op,
                        CollectiveEdgeOperation &&collective_edge_op,
                        CollectiveVertexPreprocessOperation &&collective_vertex_preprocess_op,
                        CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op,
                        bool _inner_mpi_processing);

    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation, typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
            typename CollectiveVertexPostprocessOperation>
    void advance_worker(CSRGraph &_graph,
                        FrontierCSR &_frontier,
                        EdgeOperation &&edge_op,
                        VertexPreprocessOperation &&vertex_preprocess_op,
                        VertexPostprocessOperation &&vertex_postprocess_op,
                        CollectiveEdgeOperation &&collective_edge_op,
                        CollectiveVertexPreprocessOperation &&collective_vertex_preprocess_op,
                        CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op,
                        bool _inner_mpi_processing);

    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation, typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
            typename CollectiveVertexPostprocessOperation>
    void advance_worker(CSR_VG_Graph &_graph,
                        FrontierCSR_VG &_frontier,
                        EdgeOperation &&edge_op,
                        VertexPreprocessOperation &&vertex_preprocess_op,
                        VertexPostprocessOperation &&vertex_postprocess_op,
                        CollectiveEdgeOperation &&collective_edge_op,
                        CollectiveVertexPreprocessOperation &&collective_vertex_preprocess_op,
                        CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op,
                        bool _inner_mpi_processing);

    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation, typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
            typename CollectiveVertexPostprocessOperation>
    void advance_worker(VectorCSRGraph &_graph,
                        FrontierVectorCSR &_frontier,
                        EdgeOperation &&edge_op,
                        VertexPreprocessOperation &&vertex_preprocess_op,
                        VertexPostprocessOperation &&vertex_postprocess_op,
                        CollectiveEdgeOperation &&collective_edge_op,
                        CollectiveVertexPreprocessOperation &&collective_vertex_preprocess_op,
                        CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op,
                        bool _inner_mpi_processing);

    template<typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation, typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
            typename CollectiveVertexPostprocessOperation, typename GraphContainer, typename FrontierContainer>
    void advance_worker(GraphContainer &_graph,
                        FrontierContainer &_frontier,
                        EdgeOperation &&edge_op,
                        VertexPreprocessOperation &&vertex_preprocess_op,
                        VertexPostprocessOperation &&vertex_postprocess_op,
                        CollectiveEdgeOperation &&collective_edge_op,
                        CollectiveVertexPreprocessOperation &&collective_vertex_preprocess_op,
                        CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op,
                        bool _inner_mpi_processing);
public:
    // attaches graph-processing API to the specific graph
    GraphAbstractionsGPU(VGL_Graph &_graph, TraversalDirection _initial_traversal = SCATTER);
    ~GraphAbstractionsGPU();

    // generate new frontier implementation
    // must be public since it includes device lambda
    template<typename FilterCondition>
    void generate_new_frontier_worker(CSRGraph &_graph,
                                      FrontierCSR &_frontier,
                                      FilterCondition &&filter_cond);

    template<typename FilterCondition>
    void generate_new_frontier_worker(CSR_VG_Graph &_graph,
                                      FrontierCSR_VG &_frontier,
                                      FilterCondition &&filter_cond);

    template<typename FilterCondition>
    void generate_new_frontier_worker(EdgesListGraph &_graph,
                                      FrontierEdgesList &_frontier,
                                      FilterCondition &&filter_cond);

    template<typename FilterCondition>
    void generate_new_frontier_worker(VectorCSRGraph &_graph,
                                      FrontierVectorCSR &_frontier,
                                      FilterCondition &&filter_cond);

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
    void scatter(VGL_Graph &_graph,
                 VGL_Frontier &_frontier,
                 EdgeOperation &&edge_op);

    // performs user-defined "edge_op" operation over all INCOMING edges, neighbouring specified frontier
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

    // performs user-defined "edge_op" operation over all INCOMING edges, neighbouring specified frontier
    template <typename EdgeOperation>
    void gather(VGL_Graph &_graph,
                VGL_Frontier &_frontier,
                EdgeOperation &&edge_op);

    // performs user-defined "compute_op" operation for each element in the given frontier
    template <typename ComputeOperation>
    void compute(VGL_Graph &_graph,
                 VGL_Frontier &_frontier,
                 ComputeOperation &&compute_op);

    // performs reduction using user-defined "reduce_op" operation for each element in the given frontier
    template <typename _T, typename ReduceOperation>
    _T reduce(VGL_Graph &_graph,
              VGL_Frontier &_frontier,
              ReduceOperation &&reduce_op,
              REDUCE_TYPE _reduce_type);

    // creates new frontier, which satisfy user-defined "cond" condition
    template <typename FilterCondition>
    void generate_new_frontier(VGL_Graph &_graph,
                               VGL_Frontier &_frontier,
                               FilterCondition &&filter_cond);

    friend class GraphAbstractions;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "advance_csr.hpp"
#include "graph_abstractions_gpu.hpp"
#include "compute.hpp"
#include "advance.hpp"
#include "generate_new_frontier.hpp"
#include "reduce.hpp"
#include "advance_vect_csr.hpp"
#ifdef __USE_MPI__
#include "mpi_exchange.hpp"
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
