#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../vector_register/vector_registers.h"
#include "../delayed_write/delayed_write_nec.h"

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

class GraphAbstractionsNEC
{
private:
    VectCSRGraph *processed_graph_ptr;
    TraversalDirection traversal_direction;

    template <typename ComputeOperation>
    void compute_worker(ExtendedCSRGraph &_graph,
                        FrontierNEC &_frontier,
                        ComputeOperation &&compute_op);

    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation, typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
            typename CollectiveVertexPostprocessOperation>
    void advance_worker(ExtendedCSRGraph &_graph,
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
    inline void vector_engine_per_vertex_kernel_all_active(const long long *_vertex_pointers,
                                                           const int *_adjacent_ids,
                                                           const int _first_vertex,
                                                           const int _last_vertex,
                                                           EdgeOperation edge_op,
                                                           VertexPreprocessOperation vertex_preprocess_op,
                                                           VertexPostprocessOperation vertex_postprocess_op,
                                                           const int _first_edge);

    // all-active advance inner implementation
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

    // all-active advance inner implementation
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

    // dense advance implementation
    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void vector_core_per_vertex_kernel_dense(const long long *_vertex_pointers,  const int *_adjacent_ids,
                                                    const int *_frontier_flags, const int _first_vertex,
                                                    const int _last_vertex, EdgeOperation edge_op,
                                                    VertexPreprocessOperation vertex_preprocess_op,
                                                    VertexPostprocessOperation vertex_postprocess_op,
                                                    const int _first_edge);

    // dense advance implementation
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

    // sparse advance implementation
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

    // sparse advance implementation
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
public:
    // attaches graph-processing API to the specific graph
    GraphAbstractionsNEC(VectCSRGraph &_graph, TraversalDirection _initial_traversal = SCATTER_TRAVERSAL);

    // change graph traversal direction (from GATHER to SCATTER or vice versa)
    void change_traversal_direction(TraversalDirection _new_direction);

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

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
