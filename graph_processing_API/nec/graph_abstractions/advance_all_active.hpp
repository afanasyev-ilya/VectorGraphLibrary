#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
void GraphAbstractionsNEC::vector_engine_per_vertex_kernel_all_active(const long long *_vertex_pointers,
                                                                      const int *_adjacent_ids,
                                                                      const int _first_vertex,
                                                                      const int _last_vertex,
                                                                      EdgeOperation edge_op,
                                                                      VertexPreprocessOperation vertex_preprocess_op,
                                                                      VertexPostprocessOperation vertex_postprocess_op,
                                                                      const int _first_edge)
{
    Timer tm;
    tm.start();

    DelayedWriteNEC delayed_write;
    delayed_write.init();

    long long edges_count = processed_graph_ptr->get_edges_count();
    long long direction_shift = edges_count + processed_graph_ptr->get_edges_count_in_outgoing_ve();
    int traversal = current_traversal_direction;
    int storage = CSR_STORAGE;

    for(int front_pos = _first_vertex; front_pos < _last_vertex; front_pos++)
    {
        const int src_id = front_pos;

        const long long int start = _vertex_pointers[src_id];
        const long long int end = _vertex_pointers[src_id + 1];
        const int connections_count = end - start;

        vertex_preprocess_op(src_id, connections_count, 0, delayed_write);

        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vector
        #pragma _NEC gather_reorder
        #pragma omp for schedule(static)
        for (int local_edge_pos = 0; local_edge_pos < connections_count; local_edge_pos++)
        {
            const long long internal_edge_pos = start + local_edge_pos;
            const int vector_index = get_vector_index(local_edge_pos);
            const int dst_id = _adjacent_ids[internal_edge_pos];
            const long long external_edge_pos = traversal * direction_shift + storage * edges_count + internal_edge_pos;

            edge_op(src_id, dst_id, local_edge_pos, external_edge_pos, vector_index, delayed_write);
        }

        vertex_postprocess_op(src_id, connections_count, 0, delayed_write);
    }

    tm.end();
    performance_stats.update_advance_ve_part_time(tm);
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    double work = _vertex_pointers[_last_vertex] - _vertex_pointers[_first_vertex];
    tm.print_time_and_bandwidth_stats("Advance (ve)", work, INT_ELEMENTS_PER_EDGE);
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
void GraphAbstractionsNEC::vector_core_per_vertex_kernel_all_active(const long long *_vertex_pointers,
                                                                    const int *_adjacent_ids,
                                                                    const int _first_vertex,
                                                                    const int _last_vertex,
                                                                    EdgeOperation edge_op,
                                                                    VertexPreprocessOperation vertex_preprocess_op,
                                                                    VertexPostprocessOperation vertex_postprocess_op,
                                                                    const int _first_edge)
{
    Timer tm;
    tm.start();

    DelayedWriteNEC delayed_write;
    delayed_write.init();

    long long edges_count = processed_graph_ptr->get_edges_count();
    long long direction_shift = edges_count + processed_graph_ptr->get_edges_count_in_outgoing_ve();
    int traversal = current_traversal_direction;
    int storage = CSR_STORAGE;

    #pragma omp for schedule(static, 1)
    for (int src_id = _first_vertex; src_id < _last_vertex; src_id++)
    {
        const long long int start = _vertex_pointers[src_id];
        const long long int end = _vertex_pointers[src_id + 1];
        const int connections_count = end - start;

        vertex_preprocess_op(src_id, connections_count, 0, delayed_write);

        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vector
        #pragma _NEC gather_reorder
        for (int local_edge_pos = 0; local_edge_pos < connections_count; local_edge_pos++)
        {
            const long long internal_edge_pos = start + local_edge_pos;
            const int vector_index = get_vector_index(local_edge_pos);
            const int dst_id = _adjacent_ids[internal_edge_pos];
            const long long external_edge_pos = traversal * direction_shift + storage * edges_count + internal_edge_pos;

            edge_op(src_id, dst_id, local_edge_pos, external_edge_pos, vector_index, delayed_write);
        }

        vertex_postprocess_op(src_id, connections_count, 0, delayed_write);
    }

    tm.end();
    performance_stats.update_advance_vc_part_time(tm);
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    double work = _vertex_pointers[_last_vertex] - _vertex_pointers[_first_vertex];
    tm.print_time_and_bandwidth_stats("Advance (vc)", work, INT_ELEMENTS_PER_EDGE);
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
void GraphAbstractionsNEC::ve_collective_vertex_processing_kernel_all_active(const long long *_ve_vector_group_ptrs,
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
                                                                             const int _first_edge)
{
    Timer tm;
    tm.start();

    DelayedWriteNEC delayed_write;
    delayed_write.init();

    long long edges_count = processed_graph_ptr->get_edges_count();
    long long direction_shift = edges_count + processed_graph_ptr->get_edges_count_in_outgoing_ve();
    int traversal = current_traversal_direction;
    int storage = VE_STORAGE;

    long long reg_real_start[VECTOR_LENGTH];
    int reg_real_connections_count[VECTOR_LENGTH];
    #pragma _NEC vreg(reg_real_connections_count)
    #pragma _NEC vreg(reg_real_start)

    #pragma _NEC vector
    for(int i = 0; i < VECTOR_LENGTH; i++)
    {
        reg_real_connections_count[i] = 0;
        reg_real_start[i] = 0;
    }

    #pragma omp for schedule(static, 8)
    for(int cur_vector_segment = 0; cur_vector_segment < _ve_vector_segments_count; cur_vector_segment++)
    {
        int segment_first_vertex = cur_vector_segment * VECTOR_LENGTH + _ve_starting_vertex;

        long long segment_edges_start = _ve_vector_group_ptrs[cur_vector_segment];
        int segment_connections_count = _ve_vector_group_sizes[cur_vector_segment];

        #pragma _NEC vector
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            int src_id = segment_first_vertex + i;
            reg_real_start[i] = _vertex_pointers[src_id];

            if(segment_connections_count > 0)
                reg_real_connections_count[i] = _vertex_pointers[src_id + 1] - reg_real_start[i];
            else
                reg_real_connections_count[i] = 0;

            if(src_id < _vertices_count)
                vertex_preprocess_op(src_id, reg_real_connections_count[i], i, delayed_write);
        }

        for(int edge_pos = _first_edge; edge_pos < segment_connections_count; edge_pos++)
        {
            #pragma _NEC ivdep
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            #pragma _NEC gather_reorder
            for (int i = 0; i < VECTOR_LENGTH; i++)
            {
                const int src_id = segment_first_vertex + i;

                const int vector_index = i;
                const long long internal_edge_pos = segment_edges_start + edge_pos * VECTOR_LENGTH + i;
                const int local_edge_pos = edge_pos;
                const int dst_id = _ve_adjacent_ids[internal_edge_pos];
                const long long external_edge_pos = traversal * direction_shift + storage * edges_count + internal_edge_pos;

                if((src_id < _vertices_count) && (edge_pos < reg_real_connections_count[i]))
                    edge_op(src_id, dst_id, local_edge_pos, external_edge_pos, vector_index, delayed_write);
            }
        }

        #pragma _NEC vector
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            int src_id = segment_first_vertex + i;

            if(src_id < _vertices_count)
                vertex_postprocess_op(src_id, reg_real_connections_count[i], i, delayed_write);
        }
    }

    tm.end();
    performance_stats.update_advance_collective_part_time(tm);
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    double work = _vertex_pointers[_last_vertex] - _vertex_pointers[_first_vertex];
    tm.print_time_and_bandwidth_stats("Advance (collective)", work, INT_ELEMENTS_PER_EDGE);
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
