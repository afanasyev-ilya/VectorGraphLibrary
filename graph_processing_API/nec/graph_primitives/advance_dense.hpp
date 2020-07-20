#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
void GraphPrimitivesNEC::vector_engine_per_vertex_kernel_dense(const long long *_vertex_pointers,
                                                               const int *_adjacent_ids,
                                                               const int *_frontier_flags,
                                                               const int _first_vertex,
                                                               const int _last_vertex,
                                                               EdgeOperation edge_op,
                                                               VertexPreprocessOperation vertex_preprocess_op,
                                                               VertexPostprocessOperation vertex_postprocess_op,
                                                               const int _first_edge)
{
    #ifdef __PRINT_API_PERFORMANCE_STATS__
        #pragma omp barrier
        double t1 = omp_get_wtime();
        #pragma omp barrier
    #endif

    DelayedWriteNEC delayed_write;
    delayed_write.init();

    for(int front_pos = _first_vertex; front_pos < _last_vertex; front_pos++)
    {
        const int src_id = front_pos;
        if(_frontier_flags[src_id] > 0)
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
            #pragma omp for schedule(static)
            for (int local_edge_pos = 0; local_edge_pos < connections_count; local_edge_pos++)
            {
                const long long int global_edge_pos = start + local_edge_pos;
                const int vector_index = get_vector_index(local_edge_pos);
                const int dst_id = _adjacent_ids[global_edge_pos];

                edge_op(src_id, dst_id, local_edge_pos, global_edge_pos, vector_index, delayed_write);
            }

            vertex_postprocess_op(src_id, connections_count, 0, delayed_write);
        }
    }

    #ifdef __PRINT_API_PERFORMANCE_STATS__
        #pragma omp barrier
        double t2 = omp_get_wtime();

        #pragma omp master
        {
            INNER_WALL_NEC_TIME += t2 - t1;
            INNER_ADVANCE_NEC_TIME += t2 - t1;
            DETAILED_ADVANCE_PART_1_NEC_TIME += t2 - t1;

            double work = 0;
            for(int front_pos = _first_vertex; front_pos < _last_vertex; front_pos++)
            {
                const int src_id = front_pos;
                if(_frontier_flags[src_id] > 0)
                {
                    work += _vertex_pointers[src_id + 1] - _vertex_pointers[src_id];
                }
            }
            INNER_WALL_WORK += work;
            cout << "1) time: " << (t2 - t1)*1000.0 << " ms" << endl;
            cout << "1) BW: " << sizeof(int)*INT_ELEMENTS_PER_EDGE*work/((t2-t1)*1e9) << " GB/s" << endl;
        };
        #pragma omp barrier
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
void GraphPrimitivesNEC::vector_core_per_vertex_kernel_dense(const long long *_vertex_pointers,
                                                             const int *_adjacent_ids,
                                                             const int *_frontier_flags,
                                                             const int _first_vertex,
                                                             const int _last_vertex,
                                                             EdgeOperation edge_op,
                                                             VertexPreprocessOperation vertex_preprocess_op,
                                                             VertexPostprocessOperation vertex_postprocess_op,
                                                             const int _first_edge)
{
    #ifdef __PRINT_API_PERFORMANCE_STATS__
        #pragma omp barrier
        double t1 = omp_get_wtime();
        #pragma omp barrier
    #endif

    DelayedWriteNEC delayed_write;
    delayed_write.init();

    #pragma omp for schedule(static, 1)
    for (int front_pos = _first_vertex; front_pos < _last_vertex; front_pos++)
    {
        const int src_id = front_pos;
        if(_frontier_flags[src_id] > 0)
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
                const long long int global_edge_pos = start + local_edge_pos;
                const int vector_index = get_vector_index(local_edge_pos);
                const int dst_id = _adjacent_ids[global_edge_pos];

                edge_op(src_id, dst_id, local_edge_pos, global_edge_pos, vector_index, delayed_write);
            }

            vertex_postprocess_op(src_id, connections_count, 0, delayed_write);
        }
    }

    #ifdef __PRINT_API_PERFORMANCE_STATS__
        #pragma omp barrier
        double t2 = omp_get_wtime();

        #pragma omp master
        {
            INNER_WALL_NEC_TIME += t2 - t1;
            INNER_ADVANCE_NEC_TIME += t2 - t1;
            DETAILED_ADVANCE_PART_2_NEC_TIME += t2 - t1;

            double work = 0;
            for(int front_pos = _first_vertex; front_pos < _last_vertex; front_pos++)
            {
                const int src_id = front_pos;
                if(_frontier_flags[src_id] > 0)
                {
                    work += _vertex_pointers[src_id + 1] - _vertex_pointers[src_id];
                }
            }
            INNER_WALL_WORK += work;
            cout << "2) time: " << (t2 - t1)*1000.0 << " ms" << endl;
            cout << "2) BW: " << sizeof(int)*INT_ELEMENTS_PER_EDGE*work/((t2-t1)*1e9) << " GB/s" << endl;
        };
        #pragma omp barrier
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
void GraphPrimitivesNEC::ve_collective_vertex_processing_kernel_dense(const long long *_ve_vector_group_ptrs,
                                                                      const int *_ve_vector_group_sizes,
                                                                      const int *_ve_adjacent_ids,
                                                                      const int _ve_vertices_count,
                                                                      const int _ve_starting_vertex,
                                                                      const int _ve_vector_segments_count,
                                                                      const int *_frontier_flags,
                                                                      const long long *_vertex_pointers,
                                                                      const int _first_vertex,
                                                                      const int _last_vertex,
                                                                      EdgeOperation edge_op,
                                                                      VertexPreprocessOperation vertex_preprocess_op,
                                                                      VertexPostprocessOperation vertex_postprocess_op,
                                                                      int _vertices_count,
                                                                      const int _first_edge)
{
    #ifdef __PRINT_API_PERFORMANCE_STATS__
        #pragma omp barrier
        double t1 = omp_get_wtime();
        #pragma omp barrier
    #endif

    DelayedWriteNEC delayed_write;
    delayed_write.init();

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

    #pragma omp for schedule(static, 1)
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
            #pragma _NEC sparse
            #pragma _NEC gather_reorder
            for (int i = 0; i < VECTOR_LENGTH; i++)
            {
                const int src_id = segment_first_vertex + i;

                #pragma _NEC sparse
                if(_frontier_flags[src_id] > 0)
                {
                    const int vector_index = i;
                    long long int global_edge_pos = segment_edges_start + edge_pos * VECTOR_LENGTH + i;
                    const int local_edge_pos = edge_pos;
                    const int dst_id = _ve_adjacent_ids[global_edge_pos];

                    #ifdef __USE_ADDITIONAL_EDGE_ARRAYS__
                    global_edge_pos = reg_real_start[i] + edge_pos;
                    #endif

                    if(edge_pos < reg_real_connections_count[i])
                        edge_op(src_id, dst_id, local_edge_pos, global_edge_pos, vector_index, delayed_write);
                }
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

    #ifdef __PRINT_API_PERFORMANCE_STATS__
        #pragma omp barrier
        double t2 = omp_get_wtime();

        #pragma omp master
        {
            INNER_WALL_NEC_TIME += t2 - t1;
            INNER_ADVANCE_NEC_TIME += t2 - t1;
            DETAILED_ADVANCE_PART_3_NEC_TIME += t2 - t1;

            double work = 0;
            for(int pos = 0; pos < _ve_vector_segments_count*VECTOR_LENGTH; pos++)
            {
                int src_id = pos + _ve_starting_vertex;
                int connections_count = _ve_vector_group_sizes[pos / VECTOR_LENGTH];
                if(_frontier_flags[src_id] > 0)
                {
                    work += connections_count;
                }
            }
            INNER_WALL_WORK += work;

            cout << "3) (ve) time: " << (t2 - t1)*1000.0 << " ms" << endl;
            cout << "3) (ve) BW: " << sizeof(int)*INT_ELEMENTS_PER_EDGE*work/((t2-t1)*1e9) << " GB/s" << endl;
        };
        #pragma omp barrier
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
