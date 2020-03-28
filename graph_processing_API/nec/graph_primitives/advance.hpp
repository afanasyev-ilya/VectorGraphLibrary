#pragma once

#include <set>

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation, typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
        typename CollectiveVertexPostprocessOperation >
void GraphPrimitivesNEC::advance(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                 FrontierNEC &_frontier,
                                 EdgeOperation &&edge_op,
                                 VertexPreprocessOperation &&vertex_preprocess_op,
                                 VertexPostprocessOperation &&vertex_postprocess_op,
                                 CollectiveEdgeOperation &&collective_edge_op,
                                 CollectiveVertexPreprocessOperation &&collective_vertex_preprocess_op,
                                 CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op,
                                 int _first_edge)
{
    #pragma omp barrier

    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    _frontier.print_frontier_info();
    #endif

    const long long int *vertex_pointers = outgoing_ptrs;
    const int *adjacent_ids = outgoing_ids;
    const int *ve_adjacent_ids = ve_outgoing_ids;
    int *frontier_flags = _frontier.flags;

    const int vector_engine_threshold_start = 0;
    const int vector_engine_threshold_end = _graph.get_nec_vector_engine_threshold_vertex();
    const int vector_core_threshold_start = _graph.get_nec_vector_engine_threshold_vertex();
    const int vector_core_threshold_end = _graph.get_nec_vector_core_threshold_vertex();
    const int collective_threshold_start = _graph.get_nec_vector_core_threshold_vertex();
    const int collective_threshold_end = _graph.get_vertices_count();

    if(_frontier.type == ALL_ACTIVE_FRONTIER)
    {
        if((vector_engine_threshold_end - vector_engine_threshold_start) > 0)
            vector_engine_per_vertex_kernel_all_active(vertex_pointers, adjacent_ids, vector_engine_threshold_start,
                                                       vector_engine_threshold_end, edge_op, vertex_preprocess_op,
                                                       vertex_postprocess_op, _first_edge);

        if((vector_core_threshold_end - vector_core_threshold_start) > 0)
            vector_core_per_vertex_kernel_all_active(vertex_pointers, adjacent_ids, vector_core_threshold_start,
                                                     vector_core_threshold_end, edge_op, vertex_preprocess_op,
                                                     vertex_postprocess_op, _first_edge);

        if((collective_threshold_end - collective_threshold_start) > 0)
            ve_collective_vertex_processing_kernel_all_active(ve_vector_group_ptrs, ve_vector_group_sizes,
                                                              ve_adjacent_ids, ve_vertices_count, ve_starting_vertex,
                                                              ve_vector_segments_count, collective_threshold_start, collective_threshold_end,
                                                              collective_edge_op, collective_vertex_preprocess_op,
                                                              collective_vertex_postprocess_op, vertices_count, _first_edge);
    }
    else if(_frontier.type == DENSE_FRONTIER)
    {
        if((vector_engine_threshold_end - vector_engine_threshold_start) > 0)
            vector_engine_per_vertex_kernel_dense(vertex_pointers, adjacent_ids, frontier_flags,
                                                  vector_engine_threshold_start, vector_engine_threshold_end,
                                                  edge_op, vertex_preprocess_op, vertex_postprocess_op, _first_edge);

        if((vector_core_threshold_end - vector_core_threshold_start) > 0)
            vector_core_per_vertex_kernel_dense(vertex_pointers, adjacent_ids, frontier_flags,
                                                vector_core_threshold_start, vector_core_threshold_end, edge_op,
                                                vertex_preprocess_op, vertex_postprocess_op, _first_edge);

        if((collective_threshold_end - collective_threshold_start) > 0)
            ve_collective_vertex_processing_kernel_dense(ve_vector_group_ptrs, ve_vector_group_sizes,
                                                         ve_adjacent_ids, ve_vertices_count, ve_starting_vertex, ve_vector_segments_count,
                                                         frontier_flags, collective_threshold_start, collective_threshold_end,
                                                         collective_edge_op, collective_vertex_preprocess_op,
                                                         collective_vertex_postprocess_op, vertices_count, _first_edge);
    }
    else if(_frontier.type == SPARSE_FRONTIER)
    {
        if(_frontier.vector_engine_part_size > 0)
        {
            int *frontier_ids = &(_frontier.ids[0]);
            /*vector_engine_per_vertex_kernel_dense(vertex_pointers, adjacent_ids, frontier_flags,
                                                  vector_engine_threshold_start, vector_engine_threshold_end,
                                                  edge_op, vertex_preprocess_op, vertex_postprocess_op, _first_edge);*/
            vector_engine_per_vertex_kernel_sparse(vertex_pointers, adjacent_ids, frontier_ids, _frontier.vector_engine_part_size,
                                                   edge_op, vertex_preprocess_op, vertex_postprocess_op, _first_edge);
        }

        if(_frontier.vector_core_part_size > 0)
        {
            int *frontier_ids = &(_frontier.ids[_frontier.vector_engine_part_size]);
            vector_core_per_vertex_kernel_sparse(vertex_pointers, adjacent_ids, frontier_ids, _frontier.vector_core_part_size,
                                                 edge_op, vertex_preprocess_op, vertex_postprocess_op, _first_edge);
        }

        if(_frontier.collective_part_size > 0)
        {
            int *frontier_ids = &(_frontier.ids[_frontier.vector_core_part_size + _frontier.vector_engine_part_size]);
            collective_vertex_processing_kernel_sparse(vertex_pointers, adjacent_ids, frontier_ids, _frontier.collective_part_size,
                                                       collective_threshold_start,
                                                       collective_threshold_end, collective_edge_op,
                                                       collective_vertex_preprocess_op,
                                                       collective_vertex_postprocess_op, _first_edge);
        }
    }

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    #pragma omp master
    {
        cout << endl;
    }
    #endif

    #pragma omp barrier
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
void GraphPrimitivesNEC::advance(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                 FrontierNEC &_frontier,
                                 EdgeOperation &&edge_op,
                                 VertexPreprocessOperation &&vertex_preprocess_op,
                                 VertexPostprocessOperation &&vertex_postprocess_op)
{
    advance(_graph, _frontier, edge_op, vertex_preprocess_op, vertex_postprocess_op, edge_op, vertex_preprocess_op, vertex_postprocess_op);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation>
void GraphPrimitivesNEC::advance(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                 FrontierNEC &_frontier,
                                 EdgeOperation &&edge_op)
{
    advance(_graph, _frontier, edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP, edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
