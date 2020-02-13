#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../vector_registers.h"
#include "../frontier/frontier_nec.h"
#include "../delayed_write/delayed_write_nec.h"
#include <functional>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

auto EMPTY_EDGE_OP = [] (int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos, int vector_index,
        DelayedWriteNEC &delayed_write) {};
auto EMPTY_VERTEX_OP = [] (int src_id, int connections_count, DelayedWriteNEC &delayed_write){};

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
                                             int _frontier_size);

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
                                                VertexPostprocessOperation vertex_postprocess_op, long long _edges_count);
public:
    GraphPrimitivesNEC() {};

    ~GraphPrimitivesNEC() {};

    template <typename InitOperation>
    void init(int size, InitOperation init_op);

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
                 CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_primitives_nec.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
