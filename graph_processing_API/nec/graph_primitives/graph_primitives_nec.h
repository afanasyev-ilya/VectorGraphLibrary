#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../vector_register/vector_registers.h"
#include "../frontier/frontier_nec.h"
#include "../delayed_write/delayed_write_nec.h"
#include <functional>

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

class GraphPrimitivesNEC
{
private:
    // all-active advance implementation
    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void vector_engine_per_vertex_kernel_all_active(const long long *_vertex_pointers,
                                                           const int *_adjacent_ids,
                                                           const int _first_vertex,
                                                           const int _last_vertex,
                                                           EdgeOperation edge_op,
                                                           VertexPreprocessOperation vertex_preprocess_op,
                                                           VertexPostprocessOperation vertex_postprocess_op,
                                                           const int _first_edge);

    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void vector_core_per_vertex_kernel_all_active(const long long *_vertex_pointers,
                                                         const int *_adjacent_ids,
                                                         const int _first_vertex,
                                                         const int _last_vertex,
                                                         EdgeOperation edge_op,
                                                         VertexPreprocessOperation vertex_preprocess_op,
                                                         VertexPostprocessOperation vertex_postprocess_op,
                                                         const int _first_edge);

    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void ve_collective_vertex_processing_kernel_all_active(const long long *_ve_vector_group_ptrs,
                                                                  const int *_ve_vector_group_sizes,
                                                                  const int *_ve_adjacent_ids,
                                                                  const int _ve_vertices_count,
                                                                  const int _ve_starting_vertex,
                                                                  const int _ve_vector_segments_count,
                                                                  const long long *_vertex_pointers,
                                                                  const int _first_vertex,
                                                                  const int _last_vertex,
                                                                  EdgeOperation edge_op,
                                                                  VertexPreprocessOperation vertex_preprocess_op,
                                                                  VertexPostprocessOperation vertex_postprocess_op,
                                                                  int _vertices_count,
                                                                  const int _first_edge);

    // dense advance implementation
    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void vector_engine_per_vertex_kernel_dense(const long long *_vertex_pointers, const int *_adjacent_ids,
                                                      const int *_frontier_flags, const int _first_vertex,
                                                      const int _last_vertex, EdgeOperation edge_op,
                                                      VertexPreprocessOperation vertex_preprocess_op,
                                                      VertexPostprocessOperation vertex_postprocess_op,
                                                      const int _first_edge);

    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void vector_core_per_vertex_kernel_dense(const long long *_vertex_pointers,  const int *_adjacent_ids,
                                                    const int *_frontier_flags, const int _first_vertex,
                                                    const int _last_vertex, EdgeOperation edge_op,
                                                    VertexPreprocessOperation vertex_preprocess_op,
                                                    VertexPostprocessOperation vertex_postprocess_op,
                                                    const int _first_edge);

    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void ve_collective_vertex_processing_kernel_dense(const long long *_ve_vector_group_ptrs,
                                                             const int *_ve_vector_group_sizes,
                                                             const int *_ve_adjacent_ids,
                                                             const int _ve_vertices_count,
                                                             const int _ve_starting_vertex,
                                                             const int _ve_vector_segments_count,
                                                             const int *_frontier_flags,
                                                             const long long *_vertex_pointers,
                                                             const int _first_vertex,
                                                             const int _last_vertex, EdgeOperation edge_op,
                                                             VertexPreprocessOperation vertex_preprocess_op,
                                                             VertexPostprocessOperation vertex_postprocess_op,
                                                             int _vertices_count,
                                                             const int _first_edge);

    // sparse advance implementation
    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void vector_engine_per_vertex_kernel_sparse(const long long *_vertex_pointers,
                                                       const int *_adjacent_ids,
                                                       const int *_frontier_ids,
                                                       const int _frontier_segment_size,
                                                       EdgeOperation edge_op,
                                                       VertexPreprocessOperation vertex_preprocess_op,
                                                       VertexPostprocessOperation vertex_postprocess_op,
                                                       const int _first_edge);

    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void vector_core_per_vertex_kernel_sparse(const long long *_vertex_pointers,
                                                     const int *_adjacent_ids,
                                                     const int *_frontier_ids,
                                                     const int _frontier_segment_size,
                                                     EdgeOperation edge_op,
                                                     VertexPreprocessOperation vertex_preprocess_op,
                                                     VertexPostprocessOperation vertex_postprocess_op,
                                                     const int _first_edge);

    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void collective_vertex_processing_kernel_sparse(const long long *_vertex_pointers,
                                                           const int *_adjacent_ids,
                                                           const int *_frontier_ids,
                                                           const int _frontier_size,
                                                           const int _first_vertex,
                                                           const int _last_vertex,
                                                           EdgeOperation edge_op,
                                                           VertexPreprocessOperation vertex_preprocess_op,
                                                           VertexPostprocessOperation vertex_postprocess_op,
                                                           const int _first_edge);

    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void partial_first_groups(const long long *_vertex_pointers, const int *_adjacent_ids,
                                     const int *_frontier_flags, const int _last_vertex,
                                     EdgeOperation edge_op, VertexPreprocessOperation vertex_preprocess_op,
                                     VertexPostprocessOperation vertex_postprocess_op,
                                     long long _edges_count, int _first_edge, int _last_edge);

    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void partial_last_group(const long long *_ve_vector_group_ptrs, const int *_ve_vector_group_sizes,
                                   const int *_ve_adjacent_ids, const int _ve_vertices_count,
                                   const int _ve_starting_vertex, const int _ve_vector_segments_count,
                                   const int *_frontier_flags, const int _first_vertex, const int _last_vertex,
                                   EdgeOperation edge_op, VertexPreprocessOperation vertex_preprocess_op,
                                   VertexPostprocessOperation vertex_postprocess_op,
                                   long long _edges_count, int _first_edge, int _last_edge);


    template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation, typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
            typename CollectiveVertexPostprocessOperation >
    void advance_worker(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                        FrontierNEC &_frontier,
                        EdgeOperation &&edge_op,
                        VertexPreprocessOperation &&vertex_preprocess_op,
                        VertexPostprocessOperation &&vertex_postprocess_op,
                        CollectiveEdgeOperation &&collective_edge_op,
                        CollectiveVertexPreprocessOperation &&collective_vertex_preprocess_op,
                        CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op,
                        int _first_edge = 0);

    template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation>
    void advance_worker(EdgesListGraph<_TVertexValue, _TEdgeWeight> &_graph,
                        EdgeOperation &&edge_op);

    template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation>
    void partial_advance_worker(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                FrontierNEC &_frontier,
                                EdgeOperation &&edge_op,
                                int _first_edge,
                                int _last_edge);

    // performs user-defined "compute_op" operation for each element in given frontier
    template <typename _TVertexValue, typename _TEdgeWeight, typename ComputeOperation>
    void compute_worker(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                        FrontierNEC &_frontier,
                        ComputeOperation &&compute_op);

    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void my_test(const long long *_vertex_pointers,
                        const int *_adjacent_ids,
                        const int _first_vertex,
                        const int _last_vertex,
                        EdgeOperation edge_op,
                        VertexPreprocessOperation vertex_preprocess_op,
                        VertexPostprocessOperation vertex_postprocess_op,
                        const int _first_edge);

    template <typename FilterCondition>
    int estimate_sorted_frontier_part_size(FrontierNEC &_frontier,
                                           int _first_vertex,
                                           int _last_vertex,
                                           FilterCondition &&filter_cond);

    template <typename _T, typename _TVertexValue, typename _TEdgeWeight, typename ReduceOperation>
    _T reduce_sum(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                  FrontierNEC &_frontier,
                  ReduceOperation &&reduce_op);
public:
    GraphPrimitivesNEC() {};

    ~GraphPrimitivesNEC() {};

    template <typename _TVertexValue, typename _TEdgeWeight>
    _TEdgeWeight* get_collective_weights(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                         FrontierNEC &_frontier);

    template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation, typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
            typename CollectiveVertexPostprocessOperation >
    void advance(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                 FrontierNEC &_frontier,
                 EdgeOperation &&edge_op,
                 VertexPreprocessOperation &&vertex_preprocess_op,
                 VertexPostprocessOperation &&vertex_postprocess_op,
                 CollectiveEdgeOperation &&collective_edge_op,
                 CollectiveVertexPreprocessOperation &&collective_vertex_preprocess_op,
                 CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op,
                 int _first_edge = 0);

    template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    void advance(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                 FrontierNEC &_frontier,
                 EdgeOperation &&edge_op,
                 VertexPreprocessOperation &&vertex_preprocess_op = EMPTY_VERTEX_OP,
                 VertexPostprocessOperation &&vertex_postprocess_op = EMPTY_VERTEX_OP);

    template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation>
    void advance(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                 FrontierNEC &_frontier,
                 EdgeOperation &&edge_op);

    template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation, typename Condition>
    void advance(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                 FrontierNEC &_in_frontier,
                 FrontierNEC &_out_frontier,
                 EdgeOperation &&edge_op,
                 Condition &&cond);

    template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation>
    void advance(EdgesListGraph<_TVertexValue, _TEdgeWeight> &_graph,
                 EdgeOperation &&edge_op);

    template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation>
    void partial_advance(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                         FrontierNEC &_frontier,
                         EdgeOperation &&edge_op,
                         int _first_edge,
                         int _last_edge);
    // creates new frontier, which satisfy user-defined "cond" condition
    template <typename _TVertexValue, typename _TEdgeWeight, typename Condition>
    void generate_new_frontier(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, FrontierNEC &_frontier, Condition &&cond);

    // removes elements from current frontier, which satisfy user-defined "filter_cond" condition
    template <typename _TVertexValue, typename _TEdgeWeight, typename FilterCondition>
    void filter(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, FrontierNEC &_frontier, FilterCondition &&filter_cond);

    // performs user-defined "compute_op" operation for each element in the given frontier
    template <typename _TVertexValue, typename _TEdgeWeight, typename ComputeOperation>
    void compute(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, FrontierNEC &_frontier, ComputeOperation &&compute_op);

    // performs reduction using user-defined "reduce_op" operation for each element in the given frontier
    template <typename _T, typename _TVertexValue, typename _TEdgeWeight, typename ReduceOperation>
    _T reduce(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, FrontierNEC &_frontier, ReduceOperation &&reduce_op, REDUCE_TYPE _reduce_type);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_primitives_nec.hpp"
#include "helpers.hpp"
#include "filter.hpp"
#include "compute.hpp"
#include "advance.hpp"
#include "advance_edges_list.hpp"
#include "advance_all_active.hpp"
#include "advance_dense.hpp"
#include "advance_sparse.hpp"
#include "advance_partial.hpp"
#include "generate_new_frontier.hpp"
#include "reduce.hpp"
#include "pack_array_data.hpp"
#include "tests.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
