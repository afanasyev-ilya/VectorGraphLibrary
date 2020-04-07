#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
void GraphPrimitivesNEC::partial_first_groups(const long long *_vertex_pointers,
                                              const int *_adjacent_ids,
                                              const int *_frontier_flags,
                                              const int _last_vertex,
                                              EdgeOperation edge_op,
                                              VertexPreprocessOperation vertex_preprocess_op,
                                              VertexPostprocessOperation vertex_postprocess_op,
                                              long long _edges_count,
                                              int _first_edge,
                                              int _last_edge)
{
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    double t1 = omp_get_wtime();
    #pragma omp barrier
    #endif

    long long int reg_start[VECTOR_LENGTH];
    long long int reg_end[VECTOR_LENGTH];
    int reg_connections[VECTOR_LENGTH];

    #pragma _NEC vreg(reg_start)
    #pragma _NEC vreg(reg_end)
    #pragma _NEC vreg(reg_connections)

    #pragma _NEC vector
    for(int i = 0; i < VECTOR_LENGTH; i++)
    {
        reg_start[i] = 0;
        reg_end[i] = 0;
        reg_connections[i] = 0;
    }

    DelayedWriteNEC delayed_write;
    delayed_write.init();

    #pragma omp for schedule(static, 1)
    for(int front_pos = 0; front_pos < _last_vertex; front_pos += VECTOR_LENGTH)
    {
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vector
        #pragma _NEC unroll(VECTOR_LENGTH)
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            if((front_pos + i) < _last_vertex)
            {
                int src_id = front_pos + i;
                reg_start[i] = _vertex_pointers[src_id];
                reg_end[i] = _vertex_pointers[src_id + 1];
                reg_connections[i] = reg_end[i] - reg_start[i];
            }
            else
            {
                reg_start[i] = 0;
                reg_end[i] = 0;
                reg_connections[i] = 0;
            }
        }

        int max_connections = 0;
        #pragma _NEC vector
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            if(max_connections < reg_connections[i])
            {
                max_connections = reg_connections[i];
            }
        }

        for(int edge_pos = _first_edge; edge_pos < min(max_connections, _last_edge); edge_pos++)
        {
            #pragma _NEC ivdep
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                if(((front_pos + i) < _last_vertex) && (edge_pos < reg_connections[i]))
                {
                    const int src_id = front_pos + i;
                    const int vector_index = i;
                    const long long int global_edge_pos = reg_start[i] + edge_pos;
                    const int local_edge_pos = edge_pos;
                    const int dst_id = _adjacent_ids[global_edge_pos];

                    edge_op(src_id, dst_id, local_edge_pos, global_edge_pos, vector_index, delayed_write);
                }
            }
        }
    }

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    #pragma omp barrier
    double t2 = omp_get_wtime();
    #pragma omp single
    {
        INNER_WALL_NEC_TIME += t2 - t1;
        INNER_ADVANCE_NEC_TIME += t2 - t1;

        double work = (_last_edge - _first_edge) * _last_vertex;
        cout << "partial last time: " << (t2 - t1)*1000.0 << " ms" << endl;
        cout << "partial last BW: " << work * sizeof(int) * INT_ELEMENTS_PER_EDGE/((t2 - t1)*1e9) << " GB/s" << endl;
    }

    #pragma omp barrier
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
void GraphPrimitivesNEC::partial_last_group(const long long *_ve_vector_group_ptrs,
                                            const int *_ve_vector_group_sizes,
                                            const int *_ve_adjacent_ids,
                                            const int _ve_vertices_count,
                                            const int _ve_starting_vertex,
                                            const int _ve_vector_segments_count,
                                            const int *_frontier_flags,
                                            const int _first_vertex,
                                            const int _last_vertex,
                                            EdgeOperation edge_op,
                                            VertexPreprocessOperation vertex_preprocess_op,
                                            VertexPostprocessOperation vertex_postprocess_op,
                                            long long _edges_count,
                                            int _first_edge,
                                            int _last_edge)
{
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    double t1 = omp_get_wtime();
    #pragma omp barrier
    #endif

    DelayedWriteNEC delayed_write;
    delayed_write.init();

    #pragma omp for schedule(static, 1)
    for(int cur_vector_segment = 0; cur_vector_segment < _ve_vector_segments_count; cur_vector_segment++)
    {
        int segment_first_vertex = cur_vector_segment * VECTOR_LENGTH + _ve_starting_vertex;

        long long segment_edges_start = _ve_vector_group_ptrs[cur_vector_segment];
        int segment_connections_count = _ve_vector_group_sizes[cur_vector_segment];

        for(int edge_pos = _first_edge; edge_pos < _last_edge; edge_pos++)
        {
            #pragma _NEC ivdep
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            for (int i = 0; i < VECTOR_LENGTH; i++)
            {
                const int src_id = segment_first_vertex + i;

                if((edge_pos < segment_connections_count) && (_frontier_flags[src_id] > 0))
                {
                    const int vector_index = i;
                    const long long int global_edge_pos = segment_edges_start + edge_pos * VECTOR_LENGTH + i;
                    const int local_edge_pos = edge_pos;
                    const int dst_id = _ve_adjacent_ids[global_edge_pos];

                    edge_op(src_id, dst_id, local_edge_pos, global_edge_pos, vector_index, delayed_write);
                }
            }
        }
    }

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    #pragma omp barrier
    double t2 = omp_get_wtime();
    #pragma omp single
    {
        INNER_WALL_NEC_TIME += t2 - t1;
        INNER_ADVANCE_NEC_TIME += t2 - t1;

        double work = (_last_edge - _first_edge) * _ve_vector_segments_count * VECTOR_LENGTH;
        cout << "partial last time: " << (t2 - t1)*1000.0 << " ms" << endl;
        cout << "partial last BW: " << work*sizeof(int)*INT_ELEMENTS_PER_EDGE/((t2 - t1)*1e9) << " GB/s" << endl;
    }

    #pragma omp barrier
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation>
void GraphPrimitivesNEC::partial_advance_worker(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                FrontierNEC &_frontier,
                                                EdgeOperation &&edge_op,
                                                int _first_edge,
                                                int _last_edge)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);
    const long long int *vertex_pointers = outgoing_ptrs;
    const int *adjacent_ids = outgoing_ids;
    const int *ve_adjacent_ids = ve_outgoing_ids;
    int *frontier_flags = _frontier.flags;
    int *frontier_ids = _frontier.ids;

    const int vector_core_threshold_end = _graph.get_nec_vector_core_threshold_vertex();
    const int collective_threshold_start = _graph.get_nec_vector_core_threshold_vertex();
    const int collective_threshold_end = _graph.get_vertices_count();

    partial_first_groups(vertex_pointers, adjacent_ids, frontier_flags, vector_core_threshold_end,
                         edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP, edges_count, _first_edge, _last_edge);

    partial_last_group(ve_vector_group_ptrs, ve_vector_group_sizes, ve_adjacent_ids, ve_vertices_count,
                       ve_starting_vertex, ve_vector_segments_count, frontier_flags, collective_threshold_start,
                       collective_threshold_end, edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP, edges_count, _first_edge,
                       _last_edge);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation>
void GraphPrimitivesNEC::partial_advance(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                         FrontierNEC &_frontier,
                                         EdgeOperation &&edge_op,
                                         int _first_edge,
                                         int _last_edge)
{
    if(omp_in_parallel())
    {
        #pragma omp barrier
        partial_advance_worker(_graph, _frontier, edge_op, _first_edge, _last_edge);
        #pragma omp barrier
    }
    else
    {
        #pragma omp parallel
        {
            partial_advance_worker(_graph, _frontier, edge_op, _first_edge, _last_edge);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
