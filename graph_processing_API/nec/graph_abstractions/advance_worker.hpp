#pragma once

#include <set>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation, typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
        typename CollectiveVertexPostprocessOperation>
void GraphAbstractionsNEC::advance_worker(ExtendedCSRGraph &_graph,
                                          FrontierNEC &_frontier,
                                          EdgeOperation &&edge_op,
                                          VertexPreprocessOperation &&vertex_preprocess_op,
                                          VertexPostprocessOperation &&vertex_postprocess_op,
                                          CollectiveEdgeOperation &&collective_edge_op,
                                          CollectiveVertexPreprocessOperation &&collective_vertex_preprocess_op,
                                          CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op,
                                          int _first_edge)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    _frontier.print_frontier_info(_graph);
    #pragma omp master
    {
        cout << "ADVANCE stats: " << endl;
    }
    #pragma omp barrier
    #endif

    int *frontier_flags = _frontier.flags;

    const int vector_engine_threshold_start = 0;
    const int vector_engine_threshold_end = _graph.get_vector_engine_threshold_vertex();
    const int vector_core_threshold_start = _graph.get_vector_engine_threshold_vertex();
    const int vector_core_threshold_end = _graph.get_vector_core_threshold_vertex();
    const int collective_threshold_start = _graph.get_vector_core_threshold_vertex();
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
                                                              ve_vector_segments_count, vertex_pointers, collective_threshold_start, collective_threshold_end,
                                                              collective_edge_op, collective_vertex_preprocess_op,
                                                              collective_vertex_postprocess_op, vertices_count, _first_edge);
    }
    else
    {
        if(_frontier.vector_engine_part_size > 0)
        {
            if (_frontier.vector_engine_part_type == DENSE_FRONTIER)
            {
                vector_engine_per_vertex_kernel_dense(vertex_pointers, adjacent_ids, frontier_flags,
                                                      vector_engine_threshold_start, vector_engine_threshold_end,
                                                      edge_op, vertex_preprocess_op, vertex_postprocess_op,
                                                      _first_edge);
            }
            else if (_frontier.vector_engine_part_type == SPARSE_FRONTIER)
            {
                int *frontier_ids = &(_frontier.ids[0]);
                vector_engine_per_vertex_kernel_sparse(vertex_pointers, adjacent_ids, frontier_ids,
                                                       _frontier.vector_engine_part_size,
                                                       edge_op, vertex_preprocess_op, vertex_postprocess_op,
                                                       _first_edge);
            }
        }

        if(_frontier.vector_core_part_size > 0)
        {
            if(_frontier.vector_core_part_type == DENSE_FRONTIER)
            {
                vector_core_per_vertex_kernel_dense(vertex_pointers, adjacent_ids, frontier_flags,
                                                    vector_core_threshold_start, vector_core_threshold_end, edge_op,
                                                    vertex_preprocess_op, vertex_postprocess_op, _first_edge);
            }
            else if(_frontier.vector_core_part_type == SPARSE_FRONTIER)
            {
                int *frontier_ids = &(_frontier.ids[_frontier.vector_engine_part_size]);
                vector_core_per_vertex_kernel_sparse(vertex_pointers, adjacent_ids, frontier_ids,
                                                     _frontier.vector_core_part_size,
                                                     edge_op, vertex_preprocess_op, vertex_postprocess_op, _first_edge);
            }
        }

        if(_frontier.collective_part_size > 0)
        {
            if(_frontier.collective_part_type == DENSE_FRONTIER)
            {
                ve_collective_vertex_processing_kernel_dense(ve_vector_group_ptrs, ve_vector_group_sizes,
                                                             ve_adjacent_ids, ve_vertices_count, ve_starting_vertex, ve_vector_segments_count,
                                                             frontier_flags, vertex_pointers, collective_threshold_start, collective_threshold_end,
                                                             collective_edge_op, collective_vertex_preprocess_op,
                                                             collective_vertex_postprocess_op, vertices_count, _first_edge);
            }
            else if(_frontier.collective_part_type == SPARSE_FRONTIER)
            {
                int *frontier_ids = &(_frontier.ids[_frontier.vector_core_part_size + _frontier.vector_engine_part_size]);
                collective_vertex_processing_kernel_sparse(vertex_pointers, adjacent_ids, frontier_ids, _frontier.collective_part_size,
                                                           collective_threshold_start,
                                                           collective_threshold_end, collective_edge_op,
                                                           collective_vertex_preprocess_op,
                                                           collective_vertex_postprocess_op, _first_edge);
            }
        }
    }

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    #pragma omp master
    {
        cout << endl;
    }
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
