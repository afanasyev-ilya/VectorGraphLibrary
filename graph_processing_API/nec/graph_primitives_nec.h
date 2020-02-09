#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "vector_registers.h"
#include "frontier_nec.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

auto EMPTY_OP = [] (int src_id, int connections_count){};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class GraphPrimitivesNEC
{
private:
    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    void vector_engine_per_vertex_kernel(const long long *_vertex_pointers, const int *_adjacent_ids,
                                         const int *_frontier_flags, const int _first_vertex,
                                         const int _last_vertex, EdgeOperation edge_op,
                                         VertexPreprocessOperation vertex_preprocess_op,
                                         VertexPostprocessOperation vertex_postprocess_op);

    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    void vector_core_per_vertex_kernel(const long long *_vertex_pointers,  const int *_adjacent_ids,
                                       const int *_frontier_flags, const int _first_vertex,
                                       const int _last_vertex, EdgeOperation edge_op,
                                       VertexPreprocessOperation vertex_preprocess_op,
                                       VertexPostprocessOperation vertex_postprocess_op);

    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    void collective_vertex_processing_kernel(const long long *_vertex_pointers, const int *_adjacent_ids,
                                             const int *_frontier_flags, const int _first_vertex,
                                             const int _last_vertex, EdgeOperation edge_op,
                                             VertexPreprocessOperation vertex_preprocess_op,
                                             VertexPostprocessOperation vertex_postprocess_op);

    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation>
    void ve_collective_vertex_processing_kernel(const long long *_ve_vector_group_ptrs,
                                                const int *_ve_vector_group_sizes,
                                                const int *_ve_adjacent_ids,
                                                const int _ve_vertices_count,
                                                const int _ve_starting_vertex,
                                                const int _ve_vector_segments_count,
                                                const int *_frontier_flags, const int _first_vertex,
                                                const int _last_vertex, EdgeOperation edge_op,
                                                VertexPreprocessOperation vertex_preprocess_op,
                                                VertexPostprocessOperation vertex_postprocess_op);
public:
    GraphPrimitivesNEC() {};

    ~GraphPrimitivesNEC() {};

    template <typename InitOperation>
    void init(int size, InitOperation init_op);

    template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation, typename CollectiveEdgeOperation>
    void advance(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                 FrontierNEC &_frontier,
                 EdgeOperation edge_op,
                 VertexPreprocessOperation vertex_preprocess_op,
                 VertexPostprocessOperation vertex_postprocess_op,
                 CollectiveEdgeOperation collective_edge_op,
                 bool _use_vector_extension = false);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_primitives_nec.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
