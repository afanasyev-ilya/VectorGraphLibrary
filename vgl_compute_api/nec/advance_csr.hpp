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

    for(int edge_pos = 0; edge_pos < _group_data.max_connections; edge_pos++)
    {
        #pragma omp for
        for(int vec_st = 0; vec_st < frontier_size; vec_st += VECTOR_LENGTH)
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
                int frontier_pos = vec_st + i;
                if(frontier_pos < frontier_size)
                {
                    const long long int internal_edge_pos = edge_pos * frontier_size + frontier_pos;

                    int src_id = _group_data.ids[frontier_pos];
                    int dst_id = _group_data.cell_c_adjacent_ids[internal_edge_pos];
                    if(dst_id >= 0)
                    {
                        const int vector_index = i;
                        const int local_edge_pos = edge_pos;
                        const long long external_edge_pos = _process_shift + internal_edge_pos;
                        edge_op(src_id, dst_id, local_edge_pos, external_edge_pos, vector_index);
                    }
                }
            }
        }
    }
    double t2 = omp_get_wtime();
    #pragma omp single
    {
        cout << _group_data.cell_c_size * INT_ELEMENTS_PER_EDGE*sizeof(int) / ((t2 - t1)*1e9) << " GB/s band on CELL-C" << endl;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
