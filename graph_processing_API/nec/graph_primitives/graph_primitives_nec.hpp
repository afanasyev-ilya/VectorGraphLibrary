/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
_TEdgeWeight* GraphPrimitivesNEC::get_collective_weights(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                         FrontierNEC &_frontier)
{
    if(_frontier.type() == SPARSE_FRONTIER)
        return _graph.get_outgoing_weights();
    else
        return (_graph.get_last_vertices_ve_ptr())->get_adjacent_weights();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ComputeOperation>
void GraphPrimitivesNEC::compute(ComputeOperation compute_op, int _compute_size)
{
    #pragma omp parallel for schedule(static)
    for(int vec_start = 0; vec_start < _compute_size - VECTOR_LENGTH; vec_start += VECTOR_LENGTH)
    {
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vector
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            int src_id = vec_start + i;
            compute_op(src_id);
        }
    }

    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    for(int src_id = _compute_size - VECTOR_LENGTH; src_id < _compute_size; src_id++)
    {
        compute_op(src_id);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
void GraphPrimitivesNEC::vector_engine_per_vertex_kernel(const long long *_vertex_pointers,
                                                         const int *_adjacent_ids,
                                                         const int *_frontier_flags,
                                                         const int _first_vertex,
                                                         const int _last_vertex,
                                                         EdgeOperation edge_op,
                                                         VertexPreprocessOperation vertex_preprocess_op,
                                                         VertexPostprocessOperation vertex_postprocess_op,
                                                         long long _edges_count)
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
        const int src_id = front_pos; //frontier_ids[front_pos];
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
            #pragma omp for schedule(static)
            for(int edge_pos = 0; edge_pos < connections_count; edge_pos++)
            {
                const long long int global_edge_pos = start + edge_pos;
                const int local_edge_pos = edge_pos;
                const int vector_index = edge_pos % VECTOR_LENGTH;
                int dst_id = _adjacent_ids[global_edge_pos];

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
        cout << "1) time: " << (t2 - t1)*1000.0 << " ms" << endl;
        //cout << "1) all active work: " << work << " - " << 100.0 * work/_edges_count << " %" << endl;
        cout << "1) all active BW: " << sizeof(int)*INT_ELEMENTS_PER_EDGE*work/((t2-t1)*1e9) << " GB/s" << endl;
        //cout << "1) real work: " << real_work << " - " << 100.0 * real_work/_edges_count << " %" << endl;
        cout << "1) real BW: " << sizeof(int)*INT_ELEMENTS_PER_EDGE*real_work/((t2-t1)*1e9) << " GB/s" << endl;
    };
    #pragma omp barrier
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
void GraphPrimitivesNEC::vector_core_per_vertex_kernel(const long long *_vertex_pointers,
                                                       const int *_adjacent_ids,
                                                       const int *_frontier_flags,
                                                       const int _first_vertex,
                                                       const int _last_vertex,
                                                       EdgeOperation edge_op,
                                                       VertexPreprocessOperation vertex_preprocess_op,
                                                       VertexPostprocessOperation vertex_postprocess_op,
                                                       long long _edges_count)
{
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    #pragma omp barrier
    double t1 = omp_get_wtime();
    #pragma omp barrier
    #endif

    DelayedWriteNEC delayed_write;
    delayed_write.init();

    #pragma omp for schedule(static, 8)
    for (int front_pos = _first_vertex; front_pos < _last_vertex; front_pos++)
    {
        const int src_id = front_pos;//frontier_ids[front_pos];
        if(_frontier_flags[src_id] > 0)
        {
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
        cout << "2) time: " << (t2 - t1)*1000.0 << " ms" << endl;
        //cout << "2) all active work: " << work << " - " << 100.0 * work/_edges_count << " %" << endl;
        cout << "2) all active BW: " << sizeof(int)*INT_ELEMENTS_PER_EDGE*work/((t2-t1)*1e9) << " GB/s" << endl;
        //cout << "2) real work: " << real_work << " - " << 100.0 * real_work/_edges_count << " %" << endl;
        cout << "2) real BW: " << sizeof(int)*INT_ELEMENTS_PER_EDGE*real_work/((t2-t1)*1e9) << " GB/s" << endl;
    };
    #pragma omp barrier
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
void GraphPrimitivesNEC::collective_vertex_processing_kernel(const long long *_vertex_pointers,
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
        cout << "3) all active BW: " << sizeof(int)*INT_ELEMENTS_PER_EDGE*work/((t2-t1)*1e9) << " GB/s" << endl;
        //cout << "3) real work: " << real_work << " - " << 100.0 * real_work/_edges_count << " %" << endl;
        cout << "3) real BW: " << sizeof(int)*INT_ELEMENTS_PER_EDGE*real_work/((t2-t1)*1e9) << " GB/s" << endl << endl;
    };
    #pragma omp barrier
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
void GraphPrimitivesNEC::ve_collective_vertex_processing_kernel(const long long *_ve_vector_group_ptrs,
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
                                                                int _vertices_count,
                                                                int _first_edge)
{
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    #pragma omp barrier
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

        #pragma _NEC vector
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            int src_id = segment_first_vertex + i;

            if(src_id < _vertices_count)
                vertex_preprocess_op(src_id, segment_connections_count, i, delayed_write);
        }

        for(int edge_pos = _first_edge; edge_pos < segment_connections_count; edge_pos++)
        {
            #pragma _NEC ivdep
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            for (int i = 0; i < VECTOR_LENGTH; i++)
            {
                const int src_id = segment_first_vertex + i;

                if(_frontier_flags[src_id] > 0)
                {
                    const int vector_index = i;
                    const long long int global_edge_pos = segment_edges_start + edge_pos * VECTOR_LENGTH + i;
                    const int local_edge_pos = edge_pos;
                    const int dst_id = _ve_adjacent_ids[global_edge_pos];

                    edge_op(src_id, dst_id, local_edge_pos, global_edge_pos, vector_index, delayed_write);
                }
            }
        }

        #pragma _NEC vector
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            int src_id = segment_first_vertex + i;

            if(src_id < _vertices_count)
                vertex_postprocess_op(src_id, segment_connections_count, i, delayed_write);
        }
    }

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    #pragma omp barrier
    double t2 = omp_get_wtime();
    #pragma omp master
    {
        INNER_WALL_NEC_TIME += t2 - t1;

        double work = _ve_vector_group_ptrs[_ve_vector_segments_count - 1] - _ve_vector_group_ptrs[0];
        double real_work = work;

        //cout << "3) all active work: " << work << " - " << 100.0 * work/_edges_count << " %" << endl;
        cout << "3) time: " << (t2 - t1)*1000.0 << " ms" << endl;
        cout << "3) (ve) all active BW: " << sizeof(int)*INT_ELEMENTS_PER_EDGE*work/((t2-t1)*1e9) << " GB/s" << endl;
        //cout << "3) real work: " << real_work << " - " << 100.0 * real_work/_edges_count << " %" << endl;
        cout << "3) (ve) real BW: " << sizeof(int)*INT_ELEMENTS_PER_EDGE*real_work/((t2-t1)*1e9) << " GB/s" << endl << endl;
    };
    #pragma omp barrier
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
    #pragma omp barrier

    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    const long long int *vertex_pointers = outgoing_ptrs;
    const int *adjacent_ids = outgoing_ids;
    const int *ve_adjacent_ids = ve_outgoing_ids;
    int *frontier_flags = _frontier.frontier_flags;
    int *frontier_ids = _frontier.frontier_ids;

    const int vector_engine_threshold_start = 0;
    const int vector_engine_threshold_end = _graph.get_nec_vector_engine_threshold_vertex();
    const int vector_core_threshold_start = _graph.get_nec_vector_engine_threshold_vertex();
    const int vector_core_threshold_end = _graph.get_nec_vector_core_threshold_vertex();
    const int collective_threshold_start = _graph.get_nec_vector_core_threshold_vertex();
    const int collective_threshold_end = _graph.get_vertices_count();

    vector_engine_per_vertex_kernel(vertex_pointers, adjacent_ids, frontier_flags, vector_engine_threshold_start,
                                    vector_engine_threshold_end, edge_op, vertex_preprocess_op, vertex_postprocess_op, edges_count);

    vector_core_per_vertex_kernel(vertex_pointers, adjacent_ids, frontier_flags, vector_core_threshold_start,
                                  vector_core_threshold_end, edge_op, vertex_preprocess_op, vertex_postprocess_op, edges_count);

    if(_frontier.type() == SPARSE_FRONTIER) {
        collective_vertex_processing_kernel(vertex_pointers, adjacent_ids, frontier_flags, collective_threshold_start,
                                            collective_threshold_end, collective_edge_op, collective_vertex_preprocess_op,
                                            collective_vertex_postprocess_op, edges_count,
                                            frontier_ids, _frontier.sparse_frontier_size, _first_edge);
    }
    else if(_frontier.type() == DENSE_FRONTIER) {
        ve_collective_vertex_processing_kernel(ve_vector_group_ptrs, ve_vector_group_sizes, ve_adjacent_ids,
                                               ve_vertices_count, ve_starting_vertex, ve_vector_segments_count,
                                               frontier_flags, collective_threshold_start, collective_threshold_end,
                                               collective_edge_op, collective_vertex_preprocess_op,
                                               collective_vertex_postprocess_op, edges_count, vertices_count, _first_edge);
    }

    #pragma omp barrier
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
    advance(_graph, _frontier, edge_op, vertex_preprocess_op, vertex_postprocess_op, edge_op, vertex_preprocess_op, vertex_postprocess_op);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation>
void GraphPrimitivesNEC::advance(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                 FrontierNEC &_frontier,
                                 EdgeOperation &&edge_op)
{
    advance(_graph, _frontier, edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP, edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
