/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
void GraphAbstractionsMulticore::vertex_group_advance_changed_vl(CSRVertexGroup &_group_data,
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

    LOAD_CSR_VERTEX_GROUP_DATA(_group_data);

    #pragma omp for schedule(static, 8)
    for(int idx = 0; idx < size; idx++)
    {
        int src_id = ids[idx];
        long long first = _vertex_pointers[src_id];
        long long last = _vertex_pointers[src_id + 1];
        int connections_count = last - first;

        vertex_preprocess_op(src_id, connections_count, 0);

        #pragma simd
        #pragma vector
        #pragma ivdep
        for(long long edge_pos = first; edge_pos < last; edge_pos++)
        {
            int local_edge_pos = local_edge_pos - first;
            const long long internal_edge_pos = edge_pos;
            const int vector_index = edge_pos % VECTOR_LENGTH;
            const long long external_edge_pos = _process_shift + internal_edge_pos;

            const int dst_id = _adjacent_ids[internal_edge_pos];
            edge_op(src_id, dst_id, local_edge_pos, external_edge_pos, vector_index);
        }

        vertex_postprocess_op(src_id, connections_count, 0);
    }

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.end();
    tm.print_time_and_bandwidth_stats("Advance changed vl", neighbours, INT_ELEMENTS_PER_EDGE*sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
void GraphAbstractionsMulticore::vertex_group_advance_sparse(CSRVertexGroup &_group_data,
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

    LOAD_CSR_VERTEX_GROUP_DATA(_group_data);

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
    for(int idx = 0; idx < size; idx += VECTOR_LENGTH)
    {
        #pragma simd
        #pragma vector
        #pragma ivdep
        #pragma unroll(VECTOR_LENGTH)
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            if((idx + i) < size)
            {
                int src_id = ids[idx + i];
                src_id_reg[i] = src_id;
                first_reg[i] = _vertex_pointers[src_id];
                last_reg[i] = _vertex_pointers[src_id + 1];
                connections_reg[i] = last_reg[i] - first_reg[i];
                vertex_preprocess_op(src_id, connections_reg[i], i);
            }
        }

        int max_conn = 0;
        #pragma simd
        #pragma vector
        #pragma ivdep
        #pragma unroll(VECTOR_LENGTH)
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            int conn = connections_reg[i];
            if (((idx + i) < size) && (max_conn < conn))
                max_conn = conn;
        }

        for(int pos = 0; pos < max_conn; pos++)
        {
            #pragma simd
            #pragma vector
            #pragma ivdep
            #pragma unroll(VECTOR_LENGTH)
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                long long edge_pos = first_reg[i] + pos;
                if(((idx + i) < size) && (edge_pos < last_reg[i]))
                {
                    const int src_id = ids[idx + i];

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
            if((idx + i) < size)
            {
                int src_id = ids[idx + i];
                vertex_postprocess_op(src_id, connections_reg[i], i);
            }
        }
    }

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.end();
    tm.print_time_and_bandwidth_stats("Advance sparse vl", neighbours, INT_ELEMENTS_PER_EDGE*sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
