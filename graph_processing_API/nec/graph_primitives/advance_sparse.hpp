#pragma once


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
void GraphPrimitivesNEC::vector_core_per_vertex_kernel_sparse(const long long *_vertex_pointers,
                                                              const int *_adjacent_ids,
                                                              const int *_frontier_ids,
                                                              const int *_frontier_flags,
                                                              const int _frontier_segment_size,
                                                              EdgeOperation edge_op,
                                                              VertexPreprocessOperation vertex_preprocess_op,
                                                              VertexPostprocessOperation vertex_postprocess_op)
{
    DelayedWriteNEC delayed_write;
    delayed_write.init();

    #pragma omp barrier

    #pragma omp for schedule(static, 8)
    for (int front_pos = 0; front_pos < _frontier_segment_size; front_pos++)
    {
        const int src_id = _frontier_ids[front_pos];

        const long long int start = _vertex_pointers[src_id];
        const long long int end = _vertex_pointers[src_id + 1];
        const int connections_count = end - start;

        vertex_preprocess_op(src_id, connections_count, 0, delayed_write);

        for (int edge_vec_pos = 0; edge_vec_pos < connections_count - VECTOR_LENGTH; edge_vec_pos += VECTOR_LENGTH)
        {
            #pragma _NEC ivdep
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            for (int i = 0; i < VECTOR_LENGTH; i++)
            {
                const long long int global_edge_pos = start + edge_vec_pos + i;
                const int local_edge_pos = edge_vec_pos + i;
                const int vector_index = i;
                const int dst_id = _adjacent_ids[global_edge_pos];

                edge_op(src_id, dst_id, local_edge_pos, global_edge_pos, vector_index, delayed_write);
            }
        }

        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vector
        for (int i = connections_count - VECTOR_LENGTH; i < connections_count; i++)
        {
            const long long int global_edge_pos = start + i;
            const int local_edge_pos = i;
            const int vector_index = i - (connections_count - VECTOR_LENGTH);
            const int dst_id = _adjacent_ids[global_edge_pos];

            edge_op(src_id, dst_id, local_edge_pos, global_edge_pos, vector_index, delayed_write);
        }

        vertex_postprocess_op(src_id, connections_count, 0, delayed_write);
    }

    #pragma omp barrier
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
void GraphPrimitivesNEC::collective_vertex_processing_kernel_sparse(const long long *_vertex_pointers,
                                                                    const int *_adjacent_ids,
                                                                    const int *_frontier_flags,
                                                                    const int _first_vertex,
                                                                    const int _last_vertex,
                                                                    EdgeOperation edge_op,
                                                                    VertexPreprocessOperation vertex_preprocess_op,
                                                                    VertexPostprocessOperation vertex_postprocess_op,
                                                                    long long _edges_count,
                                                                    int *_frontier_ids,
                                                                    int _frontier_size,
                                                                    int _first_edge)
{
    #ifdef __PRINT_API_PERFORMANCE_STATS__
        #pragma omp barrier
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
    for(int front_pos = 0; front_pos < _frontier_size; front_pos += VECTOR_LENGTH)
    {
        #pragma _NEC vector
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            if((front_pos + i) < _frontier_size)
            {
                int src_id = _frontier_ids[front_pos + i];
                reg_start[i] = _vertex_pointers[src_id];
                reg_end[i] = _vertex_pointers[src_id + 1];
                reg_connections[i] = reg_end[i] - reg_start[i];
                vertex_preprocess_op(src_id, reg_connections[i], i, delayed_write);
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

        for(int edge_pos = _first_edge; edge_pos < max_connections; edge_pos++)
        {
            #pragma _NEC ivdep
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                if(((front_pos + i) < _frontier_size) && (edge_pos < reg_connections[i]))
                {
                    const int src_id = _frontier_ids[front_pos + i];
                    const int vector_index = i;
                    const long long int global_edge_pos = reg_start[i] + edge_pos;
                    const int local_edge_pos = edge_pos;
                    const int dst_id = _adjacent_ids[global_edge_pos];

                    edge_op(src_id, dst_id, local_edge_pos, global_edge_pos, vector_index, delayed_write);
                }
            }
        }

        #pragma _NEC vector
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            if((front_pos + i) < _frontier_size)
            {
                int src_id = _frontier_ids[front_pos + i];
                vertex_postprocess_op(src_id, reg_connections[i], i, delayed_write);
            }
        }
    }

    #ifdef __PRINT_API_PERFORMANCE_STATS__
        #pragma omp barrier
        double t2 = omp_get_wtime();
        #pragma omp master
        {
            INNER_WALL_NEC_TIME += t2 - t1;

            double work = _vertex_pointers[_last_vertex] - _vertex_pointers[_first_vertex];
            double real_work = 0;
            for(int front_pos = _first_vertex; front_pos < _last_vertex; front_pos++)
            {
                const int src_id = front_pos;
                if(_frontier_flags[src_id] > 0)
                {
                    real_work += _vertex_pointers[src_id + 1] - _vertex_pointers[src_id];
                }
            }
            cout << "3) time: " << (t2 - t1)*1000.0 << " ms" << endl;
            //cout << "3) all active work: " << work << " - " << 100.0 * work/_edges_count << " %" << endl;
            //cout << "3) spatial BW: " << sizeof(int)*INT_ELEMENTS_PER_EDGE*work/((t2-t1)*1e9) << " GB/s" << endl;
            //cout << "3) real work: " << real_work << " - " << 100.0 * real_work/_edges_count << " %" << endl;
            cout << "3) real BW: " << sizeof(int)*INT_ELEMENTS_PER_EDGE*real_work/((t2-t1)*1e9) << " GB/s" << endl << endl;
        };
        #pragma omp barrier
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
