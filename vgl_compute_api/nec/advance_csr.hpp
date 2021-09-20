/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
void GraphAbstractionsNEC::vertex_group_advance_changed_vl(CSRVertexGroup &_group_data,
                                                           long long *_vertex_pointers,
                                                           int *_adjacent_ids,
                                                           EdgeOperation edge_op,
                                                           VertexPreprocessOperation vertex_preprocess_op,
                                                           VertexPostprocessOperation vertex_postprocess_op,
                                                           long long _process_shift)
{
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    Timer tm;
    tm.start();
    #endif

    #pragma _NEC novector
    #pragma omp for schedule(static, 8)
    for(int idx = 0; idx < _group_data.size; idx++)
    {
        int src_id = _group_data.ids[idx];
        long long start = _vertex_pointers[src_id];
        long long end = _vertex_pointers[src_id + 1];
        int connections_count = end - start;

        vertex_preprocess_op(src_id, connections_count, 0);

        #pragma _NEC novector
        for (int vec_start = 0; vec_start < connections_count; vec_start += VECTOR_LENGTH)
        {
            #pragma _NEC cncall
            #pragma _NEC ivdep
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            #pragma _NEC gather_reorder
            for (int i = 0; i < VECTOR_LENGTH; i++)
            {
                int local_edge_pos = vec_start + i;
                const long long internal_edge_pos = start + local_edge_pos;
                const long long int global_edge_pos = start + local_edge_pos;
                const int vector_index = i;
                const long long external_edge_pos = _process_shift + internal_edge_pos;

                if (local_edge_pos < connections_count)
                {
                    const int dst_id = _adjacent_ids[internal_edge_pos];
                    edge_op(src_id, dst_id, local_edge_pos, external_edge_pos, vector_index);
                }
            }
        }

        vertex_postprocess_op(src_id, connections_count, 0);
    }

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.end();
    //cout << "work size: " << _group_data.neighbours << endl;
    tm.print_time_and_bandwidth_stats("Advance vg changed vl", _group_data.neighbours, INT_ELEMENTS_PER_EDGE*sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
void GraphAbstractionsNEC::vertex_group_advance_fixed_vl(CSRVertexGroup &_group_data,
                                                         long long *_vertex_pointers,
                                                         int *_adjacent_ids,
                                                         EdgeOperation edge_op,
                                                         VertexPreprocessOperation vertex_preprocess_op,
                                                         VertexPostprocessOperation vertex_postprocess_op,
                                                         long long _process_shift)
{
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    Timer tm;
    tm.start();
    #endif

    #pragma omp for schedule(static, 8)
    for(int idx = 0; idx < _group_data.size; idx++)
    {
        int src_id = _group_data.ids[idx];
        long long first = _vertex_pointers[src_id];
        long long last = _vertex_pointers[src_id + 1];
        int connections_count = last - first;

        vertex_preprocess_op(src_id, connections_count, 0);

        #pragma _NEC cncall
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vob
        #pragma _NEC vector
        #pragma _NEC gather_reorder
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            long long edge_pos = first + i;
            if(edge_pos < last)
            {
                int local_edge_pos = i;
                const long long internal_edge_pos = edge_pos;
                const int vector_index = i;
                const long long external_edge_pos = _process_shift + internal_edge_pos;
                const int dst_id = _adjacent_ids[internal_edge_pos];
                edge_op(src_id, dst_id, local_edge_pos, external_edge_pos, vector_index);
            }
        }

        vertex_postprocess_op(src_id, connections_count, 0);
    }

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.end();
    //cout << "work size: " << _group_data.neighbours << endl;
    tm.print_time_and_bandwidth_stats("Advance vg fixed vl", _group_data.neighbours, INT_ELEMENTS_PER_EDGE*sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
void GraphAbstractionsNEC::vertex_group_advance_sparse(CSRVertexGroup &_group_data,
                                                       long long *_vertex_pointers,
                                                       int *_adjacent_ids,
                                                       EdgeOperation edge_op,
                                                       VertexPreprocessOperation vertex_preprocess_op,
                                                       VertexPostprocessOperation vertex_postprocess_op,
                                                       long long _process_shift)
{
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    Timer tm;
    tm.start();
    #endif

    int src_id_reg[VECTOR_LENGTH];
    long long first_reg[VECTOR_LENGTH];
    long long last_reg[VECTOR_LENGTH];
    int connections_reg[VECTOR_LENGTH];
    int res_reg[VECTOR_LENGTH];

    #pragma _NEC vreg(src_id_reg)
    #pragma _NEC vreg(first_reg)
    #pragma _NEC vreg(last_reg)
    #pragma _NEC vreg(connections_reg)
    #pragma _NEC vreg(res_reg)

    #pragma _NEC vector
    for(int i = 0; i < VECTOR_LENGTH; i++)
    {
        src_id_reg[i] = 0;
        first_reg[i] = 0;
        last_reg[i] = 0;
        connections_reg[i] = 0;
        res_reg[i] = 0;
    }

    #pragma omp for schedule(static, 8)
    for(int idx = 0; idx < _group_data.size; idx += VECTOR_LENGTH)
    {
        #pragma _NEC ivdep
        #pragma _NEC vector
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            if((idx + i) < _group_data.size)
            {
                int src_id = _group_data.ids[idx + i];
                src_id_reg[i] = src_id;
                first_reg[i] = _vertex_pointers[src_id];
                last_reg[i] = _vertex_pointers[src_id + 1];
                connections_reg[i] = last_reg[i] - first_reg[i];
                vertex_preprocess_op(src_id, connections_reg[i], i);
            }
        }

        int max_conn = 0;
        #pragma _NEC vector
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            int conn = connections_reg[i];
            if (((idx + i) < _group_data.size) && (max_conn < conn))
                max_conn = conn;
        }

        for(int pos = 0; pos < max_conn; pos++)
        {
            #pragma _NEC cncall
            #pragma _NEC ivdep
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vob
            #pragma _NEC vector
            #pragma _NEC gather_reorder
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                long long edge_pos = first_reg[i] + pos;
                if(((idx + i) < _group_data.size) && (edge_pos < last_reg[i]))
                {
                    const int src_id = _group_data.ids[idx + i];

                    const int vector_index = i;
                    const long long int internal_edge_pos = edge_pos;
                    const int local_edge_pos = pos;
                    const long long external_edge_pos = _process_shift + internal_edge_pos;
                    const int dst_id = _adjacent_ids[internal_edge_pos];
                    edge_op(src_id, dst_id, local_edge_pos, external_edge_pos, vector_index);
                }
            }
        }

        #pragma _NEC ivdep
        #pragma _NEC vector
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            if((idx + i) < _group_data.size)
            {
                int src_id = _group_data.ids[idx + i];
                vertex_postprocess_op(src_id, connections_reg[i], i);
            }
        }
    }

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.end();
    //cout << "work size: " << _group_data.neighbours << endl;
    tm.print_time_and_bandwidth_stats("Advance vg sparse", _group_data.neighbours, INT_ELEMENTS_PER_EDGE*sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
void GraphAbstractionsNEC::vertex_group_cell_c(CSRVertexGroupCellC &_group_data,
                                               long long *_vertex_pointers,
                                               EdgeOperation edge_op,
                                               VertexPreprocessOperation vertex_preprocess_op,
                                               VertexPostprocessOperation vertex_postprocess_op,
                                               long long _process_shift)
{
    double t1 = omp_get_wtime();
    int frontier_size = _group_data.size;

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
    for(int cur_vector_segment = 0; cur_vector_segment < _group_data.vector_segments_count; cur_vector_segment++)
    {
        int segment_first_vertex = cur_vector_segment * VECTOR_LENGTH;

        long long segment_edges_start = _group_data.vector_group_ptrs[cur_vector_segment];
        int segment_connections_count = _group_data.vector_group_sizes[cur_vector_segment];

        #pragma _NEC vector
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            int pos = segment_first_vertex + i;
            if(pos < _group_data.size)
            {
                int src_id = _group_data.vertex_ids[pos];
                reg_real_start[i] = _vertex_pointers[src_id];

                if(segment_connections_count > 0)
                    reg_real_connections_count[i] = _vertex_pointers[src_id + 1] - reg_real_start[i];
                else
                    reg_real_connections_count[i] = 0;

                vertex_preprocess_op(src_id, reg_real_connections_count[i], i);
            }
        }

        for(int edge_pos = 0; edge_pos < segment_connections_count; edge_pos++)
        {
            #pragma _NEC cncall
            #pragma _NEC ivdep
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            #pragma _NEC gather_reorder
            for (int i = 0; i < VECTOR_LENGTH; i++)
            {
                int pos = segment_first_vertex + i;
                int src_id = 0;
                if(pos < _group_data.size)
                {
                    src_id = _group_data.vertex_ids[pos];
                }

                if((pos < _group_data.size) && (edge_pos < reg_real_connections_count[i]))
                {
                    const int vector_index = i;
                    const long long internal_edge_pos = segment_edges_start + edge_pos * VECTOR_LENGTH + i;
                    const int local_edge_pos = edge_pos;
                    const int dst_id = _group_data.vector_group_adjacent_ids[internal_edge_pos];
                    const long long external_edge_pos = _group_data.old_edge_indexes[internal_edge_pos];
                    edge_op(src_id, dst_id, local_edge_pos, external_edge_pos, vector_index);
                }
            }
        }

        #pragma _NEC vector
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            int pos = segment_first_vertex + i;
            if(pos < _group_data.size)
            {
                int src_id = _group_data.vertex_ids[pos];
                vertex_postprocess_op(src_id, reg_real_connections_count[i], i);
            }
        }
    }

    double t2 = omp_get_wtime();
    #pragma omp single
    {
        cout << _group_data.edges_count_in_ve * INT_ELEMENTS_PER_EDGE*sizeof(int) / ((t2 - t1)*1e9) << " GB/s band on CELL-C" << endl;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
