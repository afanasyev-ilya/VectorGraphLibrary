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
    void reduce_worker_sum(GraphContainer &_graph,
                           FrontierContainer &_frontier,
                           ReduceOperation &&reduce_op,
                           _T &_result);

    // advance inner implementation
    template<typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation, typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
            typename CollectiveVertexPostprocessOperation>
    void advance_worker(EdgesListGraph &_graph,
                        FrontierGeneral &_frontier,
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
                        FrontierGeneral &_frontier,
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
    template<typename FilterCondition, typename GraphContainer>
    void generate_new_frontier_worker(GraphContainer &_graph, // must be public since it includes device lambda
                                      FrontierGeneral &_frontier,
                                      FilterCondition &&filter_cond);

    template<typename FilterCondition, typename GraphContainer, typename FrontierContainer> // must be public since it includes device lambda
    void generate_new_frontier_worker(GraphContainer &_graph,
                                      FrontierContainer &_frontier,
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

#include "graph_abstractions_gpu.hpp"
#include "compute.hpp"
#include "advance.hpp"
#include "generate_new_frontier.hpp"
#include "reduce.hpp"
#ifdef __USE_MPI__
#include "mpi_exchange.hpp"
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
