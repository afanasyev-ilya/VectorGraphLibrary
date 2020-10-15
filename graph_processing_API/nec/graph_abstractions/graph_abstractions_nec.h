#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../vector_register/vector_registers.h"
#include "../delayed_write/delayed_write_nec.h"
#include <cstdarg>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

auto EMPTY_EDGE_OP = [] (int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos, int vector_index,
                         DelayedWriteNEC &delayed_write) {};
auto EMPTY_VERTEX_OP = [] (int src_id, int connections_count, int vector_index, DelayedWriteNEC &delayed_write){};

auto ALL_ACTIVE_FRONTIER_CONDITION = [] (int src_id)->int
{
    return IN_FRONTIER_FLAG;
};

auto EMPTY_COMPUTE_OP = [] (int src_id, int connections_count, int vector_index) {};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class GraphAbstractionsNEC : public GraphAbstractions
{
private:
    long long direction_shift;

    // current the number of vertices, neighbouring a frontier (for Advance perf)
    long long count_frontier_neighbours(VectCSRGraph &_graph, FrontierNEC &_frontier);

    // compute inner implementation
    template <typename ComputeOperation>
    void compute_worker(UndirectedCSRGraph &_graph,
                        FrontierNEC &_frontier,
                        ComputeOperation &&compute_op);

    // reduce inner implementation
    template <typename _T, typename ReduceOperation>
    _T reduce_sum(UndirectedCSRGraph &_graph,
                  FrontierNEC &_frontier,
                  ReduceOperation &&reduce_op);

    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation, typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
            typename CollectiveVertexPostprocessOperation>
    void advance_worker(UndirectedCSRGraph &_graph,
                        FrontierNEC &_frontier,
                        EdgeOperation &&edge_op,
                        VertexPreprocessOperation &&vertex_preprocess_op,
                        VertexPostprocessOperation &&vertex_postprocess_op,
                        CollectiveEdgeOperation &&collective_edge_op,
                        CollectiveVertexPreprocessOperation &&collective_vertex_preprocess_op,
                        CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op,
                        int _first_edge);

    // all-active advance inner implementation
    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void vector_engine_per_vertex_kernel_all_active(UndirectedCSRGraph &_graph,
                                                           const int _first_vertex,
                                                           const int _last_vertex,
                                                           EdgeOperation edge_op,
                                                           VertexPreprocessOperation vertex_preprocess_op,
                                                           VertexPostprocessOperation vertex_postprocess_op,
                                                           const int _first_edge);

    // all-active advance inner implementation
    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void vector_core_per_vertex_kernel_all_active(UndirectedCSRGraph &_graph,
                                                         const int _first_vertex,
                                                         const int _last_vertex,
                                                         EdgeOperation edge_op,
                                                         VertexPreprocessOperation vertex_preprocess_op,
                                                         VertexPostprocessOperation vertex_postprocess_op,
                                                         const int _first_edge);

    // all-active advance inner implementation
    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void ve_collective_vertex_processing_kernel_all_active(UndirectedCSRGraph &_graph,
                                                                  const int _first_vertex,
                                                                  const int _last_vertex,
                                                                  EdgeOperation edge_op,
                                                                  VertexPreprocessOperation vertex_preprocess_op,
                                                                  VertexPostprocessOperation vertex_postprocess_op,
                                                                  const int _first_edge);

    // dense advance implementation
    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void vector_engine_per_vertex_kernel_dense(UndirectedCSRGraph &_graph,
                                                      FrontierNEC &_frontier,
                                                      const int _first_vertex,
                                                      const int _last_vertex,
                                                      EdgeOperation edge_op,
                                                      VertexPreprocessOperation vertex_preprocess_op,
                                                      VertexPostprocessOperation vertex_postprocess_op,
                                                      const int _first_edge);

    // dense advance implementation
    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void vector_core_per_vertex_kernel_dense(UndirectedCSRGraph &_graph,
                                                    FrontierNEC &_frontier,
                                                    const int _first_vertex,
                                                    const int _last_vertex,
                                                    EdgeOperation edge_op,
                                                    VertexPreprocessOperation vertex_preprocess_op,
                                                    VertexPostprocessOperation vertex_postprocess_op,
                                                    const int _first_edge);

    // dense advance implementation
    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void ve_collective_vertex_processing_kernel_dense(UndirectedCSRGraph &_graph,
                                                             FrontierNEC &_frontier,
                                                             const int _first_vertex,
                                                             const int _last_vertex,
                                                             EdgeOperation edge_op,
                                                             VertexPreprocessOperation vertex_preprocess_op,
                                                             VertexPostprocessOperation vertex_postprocess_op,
                                                             const int _first_edge);

    // sparse advance implementation
    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void vector_engine_per_vertex_kernel_sparse(UndirectedCSRGraph &_graph,
                                                       FrontierNEC &_frontier,
                                                       EdgeOperation edge_op,
                                                       VertexPreprocessOperation vertex_preprocess_op,
                                                       VertexPostprocessOperation vertex_postprocess_op,
                                                       const int _first_edge);

    // sparse advance implementation
    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void vector_core_per_vertex_kernel_sparse(UndirectedCSRGraph &_graph,
                                                     FrontierNEC &_frontier,
                                                     EdgeOperation edge_op,
                                                     VertexPreprocessOperation vertex_preprocess_op,
                                                     VertexPostprocessOperation vertex_postprocess_op,
                                                     const int _first_edge);

    // sparse advance implementation
    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void collective_vertex_processing_kernel_sparse(UndirectedCSRGraph &_graph,
                                                           FrontierNEC &_frontier,
                                                           const int _first_vertex,
                                                           const int _last_vertex,
                                                           EdgeOperation edge_op,
                                                           VertexPreprocessOperation vertex_preprocess_op,
                                                           VertexPostprocessOperation vertex_postprocess_op,
                                                           const int _first_edge);

    template <typename FilterCondition>
    int estimate_sorted_frontier_part_size(FrontierNEC &_frontier,
                                           long long *_vertex_pointers,
                                           int _first_vertex,
                                           int _last_vertex,
                                           FilterCondition &&filter_cond);
public:
    // attaches graph-processing API to the specific graph
    GraphAbstractionsNEC(VectCSRGraph &_graph, TraversalDirection _initial_traversal = SCATTER);
    GraphAbstractionsNEC(ShardedCSRGraph &_graph, TraversalDirection _initial_traversal = SCATTER);

    // performs user-defined "edge_op" operation over all OUTGOING edges, neighbouring specified frontier
    template <typename EdgeOperation, typename VertexPreprocessOperation, typename VertexPostprocessOperation,
            typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
            typename CollectiveVertexPostprocessOperation>
    void scatter(VectCSRGraph &_graph,
                 FrontierNEC &_frontier,
                 EdgeOperation &&edge_op,
                 VertexPreprocessOperation &&vertex_preprocess_op,
                 VertexPostprocessOperation &&vertex_postprocess_op,
                 CollectiveEdgeOperation &&collective_edge_op,
                 CollectiveVertexPreprocessOperation &&collective_vertex_preprocess_op,
                 CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op);

    // performs user-defined "edge_op" operation over all OUTGOING edges, neighbouring specified frontier
    template <typename EdgeOperation, typename VertexPreprocessOperation, typename VertexPostprocessOperation,
            typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
            typename CollectiveVertexPostprocessOperation, typename _T>
    void scatter(ShardedCSRGraph &_graph,
                 FrontierNEC &_frontier,
                 EdgeOperation &&edge_op,
                 VertexPreprocessOperation &&vertex_preprocess_op,
                 VertexPostprocessOperation &&vertex_postprocess_op,
                 CollectiveEdgeOperation &&collective_edge_op,
                 CollectiveVertexPreprocessOperation &&collective_vertex_preprocess_op,
                 CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op,
                 VerticesArray<_T> &_test_data);

    // performs user-defined "edge_op" operation over all INCOMING edges, neighbouring specified frontier
    template <typename EdgeOperation, typename VertexPreprocessOperation, typename VertexPostprocessOperation,
            typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
            typename CollectiveVertexPostprocessOperation>
    void gather(VectCSRGraph &_graph,
                FrontierNEC &_frontier,
                EdgeOperation &&edge_op,
                VertexPreprocessOperation &&vertex_preprocess_op,
                VertexPostprocessOperation &&vertex_postprocess_op,
                CollectiveEdgeOperation &&collective_edge_op,
                CollectiveVertexPreprocessOperation &&collective_vertex_preprocess_op,
                CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op);

    // performs user-defined "compute_op" operation for each element in the given frontier
    template <typename ComputeOperation>
    void compute(VectCSRGraph &_graph,
                 FrontierNEC &_frontier,
                 ComputeOperation &&compute_op);

    // performs reduction using user-defined "reduce_op" operation for each element in the given frontier
    template <typename _T, typename ReduceOperation>
    _T reduce(VectCSRGraph &_graph,
              FrontierNEC &_frontier,
              ReduceOperation &&reduce_op,
              REDUCE_TYPE _reduce_type);

    // creates new frontier, which satisfy user-defined "cond" condition
    template <typename FilterCondition>
    void generate_new_frontier(VectCSRGraph &_graph,
                               FrontierNEC &_frontier,
                               FilterCondition &&filter_cond);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "helpers.hpp"
#include "graph_abstractions_nec.hpp"
#include "compute.hpp"
#include "scatter.hpp"
#include "gather.hpp"
#include "advance_worker.hpp"
#include "advance_all_active.hpp"
#include "advance_dense.hpp"
#include "advance_sparse.hpp"
#include "generate_new_frontier.hpp"
#include "reduce.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
