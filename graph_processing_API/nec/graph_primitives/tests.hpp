/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
//#include <ftrace.h>
#endif

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
    /*int first_border = 0;
    int last_border = _last_vertex;
    int top_border = 0;
    int down_border = 0;

    int border_64 = 0;
    int border_32 = 0;
    for (int front_pos = _first_vertex; front_pos < _last_vertex - 1; front_pos++)
    {
        const int src_id = front_pos;

        const long long int start = _vertex_pointers[src_id];
        const long long int end = _vertex_pointers[src_id + 1];
        const int connections_count = end - start;

        const long long int next_start = _vertex_pointers[src_id + 1];
        const long long int next_end = _vertex_pointers[src_id + 2];
        const int next_connections_count = next_end - next_start;

        if((next_connections_count < NEC_VECTOR_CORE_THRESHOLD_VALUE) && (connections_count >= NEC_VECTOR_CORE_THRESHOLD_VALUE))
        {
            top_border = src_id;
        }

        if((next_connections_count < 8) && (connections_count >= 8))
        {
            down_border = src_id;
        }

        if((next_connections_count < 128) && (connections_count >= 128))
        {
            border_64 = src_id;
        }

        if((next_connections_count < 32) && (connections_count >= 32))
        {
            border_32 = src_id;
        }
    }

    down_border = _last_vertex;

    #pragma omp master
    {
        cout << "borders: (" << top_border << " - " << down_border << ")" << endl;
        cout << "work size: " << _vertex_pointers[down_border] - _vertex_pointers[top_border] << endl;
    }

    float reg_res[VECTOR_LENGTH];
    for(int i = 0; i < 256; i++)
        reg_res[i] = 0.0;
    #pragma _NEC vreg(reg_res)

    DelayedWriteNEC delayed_write;
    delayed_write.init();

    #pragma omp barrier
    double t1 = omp_get_wtime();
    #pragma omp barrier

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

    #pragma omp for schedule(static, 1)
    for(int front_pos = top_border; front_pos < down_border; front_pos += VECTOR_LENGTH)
    {
        #pragma _NEC vector
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            if((front_pos + i) < down_border)
            {
                int src_id = front_pos + i;//_frontier_ids[front_pos + i];
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

        if(max_connections > 0)
        {
            for (int edge_pos = _first_edge; edge_pos < max_connections; edge_pos++)
            {
                #pragma _NEC ivdep
                #pragma _NEC vovertake
                #pragma _NEC novob
                #pragma _NEC vector
                for (int i = 0; i < VECTOR_LENGTH; i++)
                {
                    if (((front_pos + i) < down_border) && (edge_pos < reg_connections[i]))
                    {
                        const int src_id = front_pos + i;//_frontier_ids[front_pos + i];
                        const int vector_index = i;
                        const long long int global_edge_pos = reg_start[i] + edge_pos;
                        const int local_edge_pos = edge_pos;
                        const int dst_id = _adjacent_ids[global_edge_pos];

                        edge_op(src_id, dst_id, local_edge_pos, global_edge_pos, vector_index, delayed_write);
                    }
                }
            }
        }
    }

    #pragma omp barrier
    double t2 = omp_get_wtime();
    #pragma omp master
    {
        INNER_WALL_NEC_TIME += t2 - t1;

        double work = _vertex_pointers[down_border] - _vertex_pointers[top_border];
        cout << "TEST BANDWIDTH: " << endl;
        cout << "TES work: " << work << endl;
        cout << "TEST 3) time: " << (t2 - t1)*1000.0 << " ms" << endl;
        cout << "MY NEW TEST 3) BW: " << sizeof(int)*5.0*work/((t2-t1)*1e9) << " GB/s" << endl << endl;
    };
    #pragma omp barrier

    /*#pragma omp barrier
    t1 = omp_get_wtime();
    #pragma omp barrier

    #pragma omp for schedule(static)
    for (int front_pos = border_64; front_pos < border_32; front_pos += 8)
    {
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vector
        for(int ij = 0; ij < 8*32; ij++)
        {
            int vertex_shift = ij / 256;
            int local_edge_pos = ij % 256;

            int src_id = front_pos + vertex_shift;
            const long long int start = _vertex_pointers[src_id];

            const long long int global_edge_pos = start + local_edge_pos;
            const int vector_index = ij;
            const int dst_id = _adjacent_ids[global_edge_pos];

            edge_op(src_id, dst_id, local_edge_pos, global_edge_pos, vector_index, delayed_write);
        }
    }

    #pragma omp barrier
    t2 = omp_get_wtime();
    #pragma omp master
    {
        INNER_WALL_NEC_TIME += t2 - t1;

        double work = 32*(border_32 - border_64);
        cout << "TEST BANDWIDTH: " << endl;
        cout << "TES work: " << work << endl;
        cout << "TEST 3) time: " << (t2 - t1)*1000.0 << " ms" << endl;
        cout << "KAZU TEST 3) BW: " << sizeof(int)*5.0*work/((t2-t1)*1e9) << " GB/s" << endl << endl;
    };
    #pragma omp barrier*/
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
