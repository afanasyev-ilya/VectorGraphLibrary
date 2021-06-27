#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_processing_API/nec/vector_register/vector_registers.h"
#include "graph_processing_API/nec/delayed_write/delayed_write_nec.h"
#include <cstdarg>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

auto EMPTY_EDGE_OP = [] (int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos, int vector_index) {};
auto EMPTY_VERTEX_OP = [] (int src_id, int connections_count, int vector_index){};

auto ALL_ACTIVE_FRONTIER_CONDITION = [] (int src_id)->int
{
    return IN_FRONTIER_FLAG;
};

auto EMPTY_COMPUTE_OP = [] __VGL_COMPUTE_ARGS__ {};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class GraphAbstractionsNEC : public GraphAbstractions
{
private:
    // current the number of vertices, neighbouring a frontier (for Advance perf)
    long long count_frontier_neighbours(VectCSRGraph &_graph, FrontierNEC &_frontier);

    bool use_safe_stores;

    long long compute_process_shift(long long _shard_shift, TraversalDirection _traversal, int _storage,
                                    long long _edges_count, bool _outgoing_graph_is_stored);

    // compute inner implementation
    template <typename ComputeOperation>
    void compute_worker(UndirectedVectCSRGraph &_graph,
                        FrontierNEC &_frontier,
                        ComputeOperation &&compute_op);

    // reduce inner implementation
    template <typename _T, typename ReduceOperation>
    _T reduce_sum(UndirectedVectCSRGraph &_graph,
                  FrontierNEC &_frontier,
                  ReduceOperation &&reduce_op);

    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation, typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
            typename CollectiveVertexPostprocessOperation>
    void advance_worker(UndirectedVectCSRGraph &_graph,
                        FrontierNEC &_frontier,
                        EdgeOperation &&edge_op,
                        VertexPreprocessOperation &&vertex_preprocess_op,
                        VertexPostprocessOperation &&vertex_postprocess_op,
                        CollectiveEdgeOperation &&collective_edge_op,
                        CollectiveVertexPreprocessOperation &&collective_vertex_preprocess_op,
                        CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op,
                        int _first_edge,
                        const long long _shard_shift,
                        bool _outgoing_graph_is_stored,
                        bool _inner_mpi_processing);

    template <typename EdgeOperation>
    void advance_worker(EdgesListGraph &_graph, EdgeOperation &&edge_op);

    // all-active advance inner implementation
    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void vector_engine_per_vertex_kernel_all_active(UndirectedVectCSRGraph &_graph,
                                                           const int _first_vertex,
                                                           const int _last_vertex,
                                                           EdgeOperation edge_op,
                                                           VertexPreprocessOperation vertex_preprocess_op,
                                                           VertexPostprocessOperation vertex_postprocess_op,
                                                           const int _first_edge,
                                                           const long long _shard_shift,
                                                           bool _outgoing_graph_is_stored);

    // all-active advance inner implementation
    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void vector_core_per_vertex_kernel_all_active(UndirectedVectCSRGraph &_graph,
                                                         const int _first_vertex,
                                                         const int _last_vertex,
                                                         EdgeOperation edge_op,
                                                         VertexPreprocessOperation vertex_preprocess_op,
                                                         VertexPostprocessOperation vertex_postprocess_op,
                                                         const int _first_edge,
                                                         const long long _shard_shift,
                                                         bool _outgoing_graph_is_stored);

    // all-active advance inner implementation
    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void ve_collective_vertex_processing_kernel_all_active(UndirectedVectCSRGraph &_graph,
                                                                  const int _first_vertex,
                                                                  const int _last_vertex,
                                                                  EdgeOperation edge_op,
                                                                  VertexPreprocessOperation vertex_preprocess_op,
                                                                  VertexPostprocessOperation vertex_postprocess_op,
                                                                  const int _first_edge,
                                                                  const long long _shard_shift,
                                                                  bool _outgoing_graph_is_stored);

    // dense advance implementation
    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void vector_engine_per_vertex_kernel_dense(UndirectedVectCSRGraph &_graph,
                                                      FrontierNEC &_frontier,
                                                      const int _first_vertex,
                                                      const int _last_vertex,
                                                      EdgeOperation edge_op,
                                                      VertexPreprocessOperation vertex_preprocess_op,
                                                      VertexPostprocessOperation vertex_postprocess_op,
                                                      const int _first_edge,
                                                      bool _outgoing_graph_is_stored);

    // dense advance implementation
    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void vector_core_per_vertex_kernel_dense(UndirectedVectCSRGraph &_graph,
                                                    FrontierNEC &_frontier,
                                                    const int _first_vertex,
                                                    const int _last_vertex,
                                                    EdgeOperation edge_op,
                                                    VertexPreprocessOperation vertex_preprocess_op,
                                                    VertexPostprocessOperation vertex_postprocess_op,
                                                    const int _first_edge,
                                                    bool _outgoing_graph_is_stored);

    // dense advance implementation
    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void ve_collective_vertex_processing_kernel_dense(UndirectedVectCSRGraph &_graph,
                                                             FrontierNEC &_frontier,
                                                             const int _first_vertex,
                                                             const int _last_vertex,
                                                             EdgeOperation edge_op,
                                                             VertexPreprocessOperation vertex_preprocess_op,
                                                             VertexPostprocessOperation vertex_postprocess_op,
                                                             const int _first_edge,
                                                             bool _outgoing_graph_is_stored);

    // sparse advance implementation
    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void vector_engine_per_vertex_kernel_sparse(UndirectedVectCSRGraph &_graph,
                                                       FrontierNEC &_frontier,
                                                       EdgeOperation edge_op,
                                                       VertexPreprocessOperation vertex_preprocess_op,
                                                       VertexPostprocessOperation vertex_postprocess_op,
                                                       const int _first_edge,
                                                       bool _outgoing_graph_is_stored);

    // sparse advance implementation
    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void vector_core_per_vertex_kernel_sparse(UndirectedVectCSRGraph &_graph,
                                                     FrontierNEC &_frontier,
                                                     EdgeOperation edge_op,
                                                     VertexPreprocessOperation vertex_preprocess_op,
                                                     VertexPostprocessOperation vertex_postprocess_op,
                                                     const int _first_edge,
                                                     bool _outgoing_graph_is_stored);

    // sparse advance implementation
    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void collective_vertex_processing_kernel_sparse(UndirectedVectCSRGraph &_graph,
                                                           FrontierNEC &_frontier,
                                                           const int _first_vertex,
                                                           const int _last_vertex,
                                                           EdgeOperation edge_op,
                                                           VertexPreprocessOperation vertex_preprocess_op,
                                                           VertexPostprocessOperation vertex_postprocess_op,
                                                           const int _first_edge,
                                                           bool _outgoing_graph_is_stored);

    template <typename FilterCondition>
    void estimate_sorted_frontier_part_size(FrontierNEC &_frontier,
                                            long long *_vertex_pointers,
                                            int _first_vertex,
                                            int _last_vertex,
                                            FilterCondition &&filter_cond,
                                            int &_elements_count,
                                            long long &_neighbours_count);
public:
    // attaches graph-processing API to the specific graph
    GraphAbstractionsNEC(VectCSRGraph &_graph, TraversalDirection _initial_traversal = SCATTER);
    GraphAbstractionsNEC(ShardedCSRGraph &_graph, TraversalDirection _initial_traversal = SCATTER);
    GraphAbstractionsNEC(EdgesListGraph &_graph, TraversalDirection _initial_traversal = ORIGINAL);

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
    template <typename EdgeOperation>
    void scatter(VectCSRGraph &_graph,
                 FrontierNEC &_frontier,
                 EdgeOperation &&edge_op);

    // performs user-defined "edge_op" operation over all OUTGOING edges, neighbouring specified frontier
    template <typename EdgeOperation, typename VertexPreprocessOperation, typename VertexPostprocessOperation,
            typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
            typename CollectiveVertexPostprocessOperation>
    void scatter(ShardedCSRGraph &_graph,
                 FrontierNEC &_frontier,
                 EdgeOperation &&edge_op,
                 VertexPreprocessOperation &&vertex_preprocess_op,
                 VertexPostprocessOperation &&vertex_postprocess_op,
                 CollectiveEdgeOperation &&collective_edge_op,
                 CollectiveVertexPreprocessOperation &&collective_vertex_preprocess_op,
                 CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op);

    // performs user-defined "edge_op" operation over all OUTGOING edges, neighbouring specified frontier
    template <typename EdgeOperation>
    void scatter(ShardedCSRGraph &_graph,
                 FrontierNEC &_frontier,
                 EdgeOperation &&edge_op);

    // performs user-defined "edge_op" operation over all OUTGOING edges, neighbouring specified frontier
    template <typename EdgeOperation>
    void scatter(EdgesListGraph &_graph,
                 EdgeOperation &&edge_op);

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

    // performs user-defined "edge_op" operation over all INCOMING edges, neighbouring specified frontier
    template <typename EdgeOperation>
    void gather(VectCSRGraph &_graph,
                FrontierNEC &_frontier,
                EdgeOperation &&edge_op);

    // performs user-defined "edge_op" operation over all INCOMING edges, neighbouring specified frontier
    template <typename EdgeOperation, typename VertexPreprocessOperation, typename VertexPostprocessOperation,
            typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
            typename CollectiveVertexPostprocessOperation>
    void gather(ShardedCSRGraph &_graph,
                FrontierNEC &_frontier,
                EdgeOperation &&edge_op,
                VertexPreprocessOperation &&vertex_preprocess_op,
                VertexPostprocessOperation &&vertex_postprocess_op,
                CollectiveEdgeOperation &&collective_edge_op,
                CollectiveVertexPreprocessOperation &&collective_vertex_preprocess_op,
                CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op);

    // performs user-defined "edge_op" operation over all INCOMING edges, neighbouring specified frontier
    template <typename EdgeOperation>
    void gather(ShardedCSRGraph &_graph,
                FrontierNEC &_frontier,
                EdgeOperation &&edge_op);

    // performs user-defined "compute_op" operation for each element in the given frontier
    template <typename ComputeOperation>
    void compute(VectCSRGraph &_graph,
                 FrontierNEC &_frontier,
                 ComputeOperation &&compute_op);

    // performs user-defined "compute_op" operation for each element in the given frontier
    template <typename ComputeOperation>
    void compute(ShardedCSRGraph &_graph,
                 FrontierNEC &_frontier,
                 ComputeOperation &&compute_op);

    // performs reduction using user-defined "reduce_op" operation for each element in the given frontier
    template <typename _T, typename ReduceOperation>
    _T reduce(VectCSRGraph &_graph,
              FrontierNEC &_frontier,
              ReduceOperation &&reduce_op,
              REDUCE_TYPE _reduce_type);

    // performs reduction using user-defined "reduce_op" operation for each element in the given frontier
    template <typename _T, typename ReduceOperation>
    _T reduce(ShardedCSRGraph &_graph,
              FrontierNEC &_frontier,
              ReduceOperation &&reduce_op,
              REDUCE_TYPE _reduce_type);

    // creates new frontier, which satisfy user-defined "cond" condition
    template <typename FilterCondition>
    void generate_new_frontier(VectCSRGraph &_graph,
                               FrontierNEC &_frontier,
                               FilterCondition &&filter_cond);

    template <typename _T1, typename _T2>
    void pack_vertices_arrays(VerticesArray<VGL_PACK_TYPE> &_packed_data,
                              VerticesArray<_T1> &_first,
                              VerticesArray<_T2> &_second);

    template <typename _T1, typename _T2>
    void unpack_vertices_arrays(VerticesArray<VGL_PACK_TYPE> &_packed_data,
                                VerticesArray<_T1> &_first,
                                VerticesArray<_T2> &_second);

    void enable_safe_stores() {use_safe_stores = true;};
    void disable_safe_stores() {use_safe_stores = false;};

    #ifdef __USE_MPI__
    template <typename _TGraph, typename _T>
    void exchange_vertices_array(DataExchangePolicy _policy, _TGraph &_graph, VerticesArray<_T> &_data);

    template <typename _TGraph, typename _T, typename MergeOp>
    void exchange_vertices_array(DataExchangePolicy _policy, _TGraph &_graph, VerticesArray<_T> &_data,
                                 MergeOp &&_merge_op);

    template <typename _TGraph, typename _T, typename MergeOp>
    void exchange_vertices_array(DataExchangePolicy _policy, _TGraph &_graph, VerticesArray<_T> &_data,
                                 VerticesArray<_T> &_old_data, MergeOp &&_merge_op);
    #endif
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "pack.hpp"
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
#ifdef __USE_MPI__
#include "mpi_exchange.hpp"
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
