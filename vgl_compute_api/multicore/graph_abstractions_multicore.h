#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "vector_register/vector_registers.h"
#include <cstdarg>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class GraphAbstractionsMulticore : public GraphAbstractions
{
private:
    bool use_safe_stores;

    // compute inner implementation
    template <typename ComputeOperation, typename GraphContainer, typename FrontierContainer>
    void compute_worker(GraphContainer &_graph, FrontierContainer &_frontier, ComputeOperation &&compute_op);

    // reduce inner implementation
    template <typename _T, typename ReduceOperation, typename GraphContainer, typename FrontierContainer>
    void reduce_worker(GraphContainer &_graph, FrontierContainer &_frontier, ReduceOperation &&reduce_op,
                       REDUCE_TYPE _reduce_type, _T &_result);

    template <typename _T, typename ReduceOperation, typename GraphContainer, typename FrontierContainer>
    void reduce_worker_sum(GraphContainer &_graph, FrontierContainer &_frontier, ReduceOperation &&reduce_op,
                           _T &_result);

    template <typename _T, typename ReduceOperation, typename GraphContainer, typename FrontierContainer>
    void reduce_worker_max(GraphContainer &_graph, FrontierContainer &_frontier, ReduceOperation &&reduce_op,
                           _T &_result);

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

    template <typename EdgeOperation, typename VertexPreprocessOperation,
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

    template <typename FilterCondition>
    void generate_new_frontier_worker(CSRGraph &_graph,
                                      FrontierCSR &_frontier,
                                      FilterCondition &&filter_cond);

    template <typename FilterCondition>
    void generate_new_frontier_worker(CSR_VG_Graph &_graph,
                                      FrontierCSR_VG &_frontier,
                                      FilterCondition &&filter_cond);

    template <typename FilterCondition>
    void generate_new_frontier_worker(EdgesListGraph &_graph,
                                      FrontierEdgesList &_frontier,
                                      FilterCondition &&filter_cond);

    template <typename FilterCondition>
    void generate_new_frontier_worker(VectorCSRGraph &_graph,
                                      FrontierVectorCSR &_frontier,
                                      FilterCondition &&filter_cond);

    // all-active advance inner implementation
    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void vector_engine_per_vertex_kernel_all_active(VectorCSRGraph &_graph,
                                                           const int _first_vertex,
                                                           const int _last_vertex,
                                                           EdgeOperation edge_op,
                                                           VertexPreprocessOperation vertex_preprocess_op,
                                                           VertexPostprocessOperation vertex_postprocess_op);

    // all-active advance inner implementation
    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void vector_core_per_vertex_kernel_all_active(VectorCSRGraph &_graph,
                                                         const int _first_vertex,
                                                         const int _last_vertex,
                                                         EdgeOperation edge_op,
                                                         VertexPreprocessOperation vertex_preprocess_op,
                                                         VertexPostprocessOperation vertex_postprocess_op);

    // all-active advance inner implementation
    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void ve_collective_vertex_processing_kernel_all_active(VectorCSRGraph &_graph,
                                                                  const int _first_vertex,
                                                                  const int _last_vertex,
                                                                  EdgeOperation edge_op,
                                                                  VertexPreprocessOperation vertex_preprocess_op,
                                                                  VertexPostprocessOperation vertex_postprocess_op);

    // dense advance implementation
    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void vector_engine_per_vertex_kernel_dense(VectorCSRGraph &_graph,
                                                      FrontierVectorCSR &_frontier,
                                                      const int _first_vertex,
                                                      const int _last_vertex,
                                                      EdgeOperation edge_op,
                                                      VertexPreprocessOperation vertex_preprocess_op,
                                                      VertexPostprocessOperation vertex_postprocess_op);

    // dense advance implementation
    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void vector_core_per_vertex_kernel_dense(VectorCSRGraph &_graph,
                                                    FrontierVectorCSR &_frontier,
                                                    const int _first_vertex,
                                                    const int _last_vertex,
                                                    EdgeOperation edge_op,
                                                    VertexPreprocessOperation vertex_preprocess_op,
                                                    VertexPostprocessOperation vertex_postprocess_op);

    // dense advance implementation
    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void ve_collective_vertex_processing_kernel_dense(VectorCSRGraph &_graph,
                                                             FrontierVectorCSR &_frontier,
                                                             const int _first_vertex,
                                                             const int _last_vertex,
                                                             EdgeOperation edge_op,
                                                             VertexPreprocessOperation vertex_preprocess_op,
                                                             VertexPostprocessOperation vertex_postprocess_op);

    // sparse advance implementation
    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void vector_engine_per_vertex_kernel_sparse(VectorCSRGraph &_graph,
                                                       FrontierVectorCSR &_frontier,
                                                       EdgeOperation edge_op,
                                                       VertexPreprocessOperation vertex_preprocess_op,
                                                       VertexPostprocessOperation vertex_postprocess_op);

    // sparse advance implementation
    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void vector_core_per_vertex_kernel_sparse(VectorCSRGraph &_graph,
                                                     FrontierVectorCSR &_frontier,
                                                     EdgeOperation edge_op,
                                                     VertexPreprocessOperation vertex_preprocess_op,
                                                     VertexPostprocessOperation vertex_postprocess_op);

    // sparse advance implementation
    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void collective_vertex_processing_kernel_sparse(VectorCSRGraph &_graph,
                                                           FrontierVectorCSR &_frontier,
                                                           const int _first_vertex,
                                                           const int _last_vertex,
                                                           EdgeOperation edge_op,
                                                           VertexPreprocessOperation vertex_preprocess_op,
                                                           VertexPostprocessOperation vertex_postprocess_op);

    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void vertex_group_advance_changed_vl(CSRVertexGroup &_group_data,
                                                long long *_vertex_pointers,
                                                int *_adjacent_ids,
                                                EdgeOperation edge_op,
                                                VertexPreprocessOperation vertex_preprocess_op,
                                                VertexPostprocessOperation vertex_postprocess_op,
                                                long long _process_shift);

    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void vertex_group_advance_sparse(CSRVertexGroup &_group_data,
                                            long long *_vertex_pointers,
                                            int *_adjacent_ids,
                                            EdgeOperation edge_op,
                                            VertexPreprocessOperation vertex_preprocess_op,
                                            VertexPostprocessOperation vertex_postprocess_op,
                                            long long _process_shift);

    template <typename FilterCondition>
    void estimate_sorted_frontier_part_size(FrontierVectorCSR &_frontier,
                                            long long *_vertex_pointers,
                                            int _first_vertex,
                                            int _last_vertex,
                                            FilterCondition &&filter_cond,
                                            int &_elements_count,
                                            long long &_neighbours_count);
public:
    // attaches graph-processing API to the specific graph
    GraphAbstractionsMulticore(VGL_Graph &_graph, TraversalDirection _initial_traversal = SCATTER);

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

    /*template <typename _T1, typename _T2>
    void pack_vertices_arrays(VerticesArray<VGL_PACK_TYPE> &_packed_data,
                              VerticesArray<_T1> &_first,
                              VerticesArray<_T2> &_second);

    template <typename _T1, typename _T2>
    void unpack_vertices_arrays(VerticesArray<VGL_PACK_TYPE> &_packed_data,
                                VerticesArray<_T1> &_first,
                                VerticesArray<_T2> &_second);*/

    void enable_safe_stores() {use_safe_stores = true;};
    void disable_safe_stores() {use_safe_stores = false;};

    friend class GraphAbstractions;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//#include "pack.hpp"
#include "helpers.hpp"
#include "graph_abstractions_multicore.hpp"
#include "compute.hpp"
#include "scatter.hpp"
#include "gather.hpp"
#include "advance_worker.hpp"
#include "advance_all_active.hpp"
#include "advance_dense.hpp"
#include "advance_sparse.hpp"
#include "advance_csr.hpp"
#include "generate_new_frontier.hpp"
#include "reduce.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
