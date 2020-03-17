#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../vector_register/vector_registers.h"
#include "../frontier/frontier_nec.h"
#include "../delayed_write/delayed_write_nec.h"
#include <functional>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 5.0

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

auto EMPTY_EDGE_OP = [] (int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos, int vector_index,
        DelayedWriteNEC &delayed_write) {};
auto EMPTY_VERTEX_OP = [] (int src_id, int connections_count, int vector_index, DelayedWriteNEC &delayed_write){};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double INNER_WALL_NEC_TIME = 0;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class GraphPrimitivesNEC
{
private:
    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void vector_engine_per_vertex_kernel(const long long *_vertex_pointers, const int *_adjacent_ids,
                                         const int *_frontier_flags, const int _first_vertex,
                                         const int _last_vertex, EdgeOperation edge_op,
                                         VertexPreprocessOperation vertex_preprocess_op,
                                         VertexPostprocessOperation vertex_postprocess_op, long long _edges_count);

    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void vector_core_per_vertex_kernel(const long long *_vertex_pointers,  const int *_adjacent_ids,
                                       const int *_frontier_flags, const int _first_vertex,
                                       const int _last_vertex, EdgeOperation edge_op,
                                       VertexPreprocessOperation vertex_preprocess_op,
                                       VertexPostprocessOperation vertex_postprocess_op, long long _edges_count);

    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void collective_vertex_processing_kernel(const long long *_vertex_pointers, const int *_adjacent_ids,
                                             const int *_frontier_flags, const int _first_vertex,
                                             const int _last_vertex, EdgeOperation edge_op,
                                             VertexPreprocessOperation vertex_preprocess_op,
                                             VertexPostprocessOperation vertex_postprocess_op, long long _edges_count,
                                             int *_frontier_ids,
                                             int _frontier_size,
                                             int _first_edge);

    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    inline void ve_collective_vertex_processing_kernel(const long long *_ve_vector_group_ptrs,
                                                       const int *_ve_vector_group_sizes,
                                                       const int *_ve_adjacent_ids,
                                                       const int _ve_vertices_count,
                                                       const int _ve_starting_vertex,
                                                       const int _ve_vector_segments_count,
                                                       const int *_frontier_flags, const int _first_vertex,
                                                       const int _last_vertex, EdgeOperation edge_op,
                                                       VertexPreprocessOperation vertex_preprocess_op,
                                                       VertexPostprocessOperation vertex_postprocess_op,
                                                       long long _edges_count, int _vertices_count,
                                                       int _first_edge);

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
public:
    GraphPrimitivesNEC() {};

    ~GraphPrimitivesNEC() {};

    template <typename _TVertexValue, typename _TEdgeWeight>
    _TEdgeWeight* get_collective_weights(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                         FrontierNEC &_frontier);

    template <typename ComputeOperation>
    void compute(ComputeOperation compute_op, int _compute_size);

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

    template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation>
    void partial_advance(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                         FrontierNEC &_frontier,
                         EdgeOperation &&edge_op,
                         int _first_edge,
                         int _last_edge);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_primitives_nec.hpp"
#include "partial_advance.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
