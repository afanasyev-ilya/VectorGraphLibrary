#pragma once

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
public:
    // attaches graph-processing API to the specific graph
    GraphAbstractionsNEC(VectCSRGraph &_graph,
                         TraversalDirection _initial_traversal = SCATTER_TRAVERSAL);

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

    // performs user-defined "compute_op" operation for each element in the given frontier
    template <typename ComputeOperation>
    void compute(VectCSRGraph &_graph,
                 FrontierNEC &_frontier,
                 ComputeOperation &&compute_op);
};


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_abstractions_nec.hpp"
#include "compute.hpp"
#include "scatter.hpp"
#include "advance_worker.hpp"
#include "advance_all_active.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
