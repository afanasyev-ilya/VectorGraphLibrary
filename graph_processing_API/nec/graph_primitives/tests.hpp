/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
void GraphPrimitivesNEC::my_test(const long long *_vertex_pointers, const int *_adjacent_ids,
                                 const int _first_vertex,
                                 const int _last_vertex,
                                 EdgeOperation edge_op,
                                 VertexPreprocessOperation vertex_preprocess_op,
                                 VertexPostprocessOperation vertex_postprocess_op,
                                 const int _first_edge)
{
    int small_border = 0;
    for (int front_pos = _first_vertex; front_pos < _last_vertex - 1; front_pos++)
    {
        const int src_id = front_pos;

        const long long int start = _vertex_pointers[src_id];
        const long long int end = _vertex_pointers[src_id + 1];
        const int connections_count = end - start;

        const long long int next_start = _vertex_pointers[src_id + 1];
        const long long int next_end = _vertex_pointers[src_id + 2];
        const int next_connections_count = next_end - next_start;

        if((next_connections_count < VECTOR_LENGTH/2) && (connections_count >= VECTOR_LENGTH/2))
        {
            small_border = src_id;
            break;
        }
    }

    #pragma omp master
    {
        cout << "small_border: (" << _first_vertex << ", " << small_border << ")" << endl;
    }

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    #pragma omp barrier
    double t1 = omp_get_wtime();
    #pragma omp barrier
    #endif

    DelayedWriteNEC delayed_write;
    delayed_write.init();

    int reg_shifts[VECTOR_LENGTH];
    int reg_gather[VECTOR_LENGTH];
    int reg_res[VECTOR_LENGTH];
    int reg_src_ids[VECTOR_LENGTH];
    int reg_start[VECTOR_LENGTH];

    #pragma _NEC vreg(reg_shifts)
    #pragma _NEC vreg(reg_gather)
    #pragma _NEC vreg(reg_res)
    #pragma _NEC vreg(reg_src_ids)
    #pragma _NEC vreg(reg_start)

    #pragma _NEC vector
    for(int i = 0; i < VECTOR_LENGTH; i++)
    {
        if(i < 128)
            reg_shifts[i] = i;
        else
            reg_shifts[i] = i - 128;
        reg_gather[i] = 0;
        reg_res[i] = 0;
        reg_src_ids[i] = 0;
        reg_start[i] = 0;
    }

    #pragma omp for schedule(static, 8)
    for (int front_pos = _first_vertex; front_pos < small_border; front_pos += 2)
    {
        int src_id_1 = front_pos;
        int src_id_2 = front_pos + 1;

        const long long int start_1 = _vertex_pointers[src_id_1];
        const long long int end_1 = _vertex_pointers[src_id_1 + 1];
        const int connections_count_1 = end_1 - start_1;

        const long long int start_2 = _vertex_pointers[src_id_2];
        const long long int end_2 = _vertex_pointers[src_id_2 + 1];
        const int connections_count_2 = end_2 - start_2;

        #pragma _NEC vector
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            if(i < 128)
                reg_src_ids[i] = src_id_1;
            else
                reg_src_ids[i] = src_id_2;
        }

        #pragma _NEC vector
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            if(i < 128)
                reg_start[i] = start_1;
            else
                reg_start[i] = start_2;
        }

        #pragma _NEC vector
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            reg_gather[i] = reg_start[i] + reg_shifts[i];
        }

        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vector
        for (int i = 0; i < VECTOR_LENGTH; i++)
        {
            int dst_id = _adjacent_ids[reg_gather[i]];
            edge_op(reg_src_ids[i], dst_id, 0, 0, 0, delayed_write);
        }

        /*int src_id = front_pos;

        const long long int start = _vertex_pointers[src_id];
        const long long int end = _vertex_pointers[src_id + 1];
        const int connections_count = end - start;

        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vector
        for (int i = 0; i < VECTOR_LENGTH; i++)
        {
            const int local_edge_pos = i;
            const long long int global_edge_pos = start + local_edge_pos;

            const int vector_index = i;
            const int dst_id = _adjacent_ids[global_edge_pos];

            edge_op(src_id, dst_id, local_edge_pos, global_edge_pos, vector_index, delayed_write);
        }*/
    }

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    #pragma omp barrier
        double t2 = omp_get_wtime();
        #pragma omp master
        {
            INNER_WALL_NEC_TIME += t2 - t1;

            double work = _vertex_pointers[small_border] - _vertex_pointers[_first_vertex];
            cout << "3) time: " << (t2 - t1)*1000.0 << " ms" << endl;
            cout << "3) BW: " << sizeof(int)*INT_ELEMENTS_PER_EDGE*work/((t2-t1)*1e9) << " GB/s" << endl;
        };
        #pragma omp barrier
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
