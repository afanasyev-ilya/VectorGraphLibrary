#pragma once

#include <set>

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation, typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
        typename CollectiveVertexPostprocessOperation >
void GraphPrimitivesNEC::advance_worker(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
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
    if(omp_in_parallel())
    {
        #pragma omp barrier
        advance_worker(_graph, _frontier, edge_op, vertex_preprocess_op, vertex_postprocess_op,
                       collective_edge_op, collective_vertex_preprocess_op, collective_vertex_postprocess_op, _first_edge);
        #pragma omp barrier
    }
    else
    {
        #pragma omp parallel
        {
            advance_worker(_graph, _frontier, edge_op, vertex_preprocess_op, vertex_postprocess_op,
                           collective_edge_op, collective_vertex_preprocess_op, collective_vertex_postprocess_op, _first_edge);
        }
    }
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
    if(omp_in_parallel())
    {
        #pragma omp barrier
        advance_worker(_graph, _frontier, edge_op, vertex_preprocess_op, vertex_postprocess_op, edge_op,
                       vertex_preprocess_op, vertex_postprocess_op);
        #pragma omp barrier
    }
    else
    {
        #pragma omp parallel
        {
            advance_worker(_graph, _frontier, edge_op, vertex_preprocess_op, vertex_postprocess_op, edge_op,
                           vertex_preprocess_op, vertex_postprocess_op);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation>
void GraphPrimitivesNEC::advance(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                 FrontierNEC &_frontier,
                                 EdgeOperation &&edge_op)
{
    if(omp_in_parallel())
    {
        #pragma omp barrier
        advance_worker(_graph, _frontier, edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP, edge_op, EMPTY_VERTEX_OP,
                       EMPTY_VERTEX_OP);
        #pragma omp barrier
    }
    else
    {
        #pragma omp parallel
        {
            advance_worker(_graph, _frontier, edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP, edge_op, EMPTY_VERTEX_OP,
                           EMPTY_VERTEX_OP);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation, typename Condition>
void GraphPrimitivesNEC::advance(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                 FrontierNEC &_in_frontier,
                                 FrontierNEC &_out_frontier,
                                 EdgeOperation &&edge_op,
                                 Condition &&cond)
{
    #pragma omp parallel
    {
        advance_worker(_graph, _in_frontier, edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP, edge_op, EMPTY_VERTEX_OP,
                       EMPTY_VERTEX_OP);
    }

    int vertices_count = _graph.get_vertices_count();
    int old_frontier_size = _in_frontier.size();

    if((old_frontier_size > VECTOR_LENGTH) || (old_frontier_size == 1))
    {
        generate_new_frontier(_graph, _out_frontier, cond);
    }
    else
    {
        LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

        #ifdef __PRINT_API_PERFORMANCE_STATS__
        double t1 = omp_get_wtime();
        #endif

        int cur_pos = 0;
        int *tmp_frontier = _in_frontier.work_buffer;

        int reg_connections[VECTOR_LENGTH];
        int reg_shifts[VECTOR_LENGTH];

        #pragma _NEC vector
        for(int i = 0; i < _in_frontier.current_size; i++)
        {
            int src_id = _in_frontier.ids[i];
            const long long int start = outgoing_ptrs[src_id];
            const long long int end = outgoing_ptrs[src_id + 1];
            const int connections_count = end - start;
            reg_connections[i] = connections_count;
            reg_shifts[i] = 0;
        }

        #pragma _NEC vector
        for(int i = 1; i < _in_frontier.current_size; i++)
        {
            reg_shifts[i] = reg_shifts[i - 1] + reg_connections[i - 1];
        }

        #pragma omp parallel for
        for(int front_pos = 0; front_pos < _in_frontier.current_size; front_pos++)
        {
            int src_id = _in_frontier.ids[front_pos];
            const long long int start = outgoing_ptrs[src_id];

            #pragma _NEC ivdep
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            for (int local_edge_pos = 0; local_edge_pos < reg_connections[front_pos]; local_edge_pos++)
            {
                const long long int global_edge_pos = start + local_edge_pos;
                const int dst_id = outgoing_ids[global_edge_pos];
                if(cond(dst_id))
                {
                    tmp_frontier[local_edge_pos + reg_shifts[front_pos]] = dst_id;
                }
                else
                {
                    tmp_frontier[local_edge_pos + reg_shifts[front_pos]] = -1;
                }
            }
        }

        int tmp_frontier_size = reg_shifts[_in_frontier.current_size - 1] + reg_connections[_in_frontier.current_size - 1];

        int new_frontier_size = dense_copy_if2(tmp_frontier, tmp_frontier, _in_frontier.ids, tmp_frontier_size);

        _out_frontier.current_size = new_frontier_size;
        _out_frontier.type = SPARSE_FRONTIER;
        _out_frontier.vector_engine_part_type = SPARSE_FRONTIER;
        _out_frontier.vector_core_part_type = SPARSE_FRONTIER;
        _out_frontier.collective_part_type = SPARSE_FRONTIER;
        _out_frontier.vector_engine_part_size = 0;
        _out_frontier.vector_core_part_size = 0;
        _out_frontier.collective_part_size = new_frontier_size;

        #pragma _NEC ivdep
        #pragma _NEC vector
        #pragma omp parallel for
        for(int i = 0; i < _out_frontier.current_size; i++)
        {
            _out_frontier.ids[i] = tmp_frontier[i];
        }

        #ifdef __PRINT_API_PERFORMANCE_STATS__
        double t2 = omp_get_wtime();
        cout << "SMALL GNF time: " << 1000.0*(t2 - t1) << " ms" << endl;
        cout << "SMALL GNF BW: " << 2.0*sizeof(int)*tmp_frontier_size << " GB/s" << endl;
        #endif
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
