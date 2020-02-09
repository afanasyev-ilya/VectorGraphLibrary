/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename InitOperation>
void GraphPrimitivesNEC::init(int _size, InitOperation init_op)
{
    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma omp for schedule(static)
    for(int src_id = 0; src_id < _size; src_id++)
    {
        init_op(src_id);
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
                                                         VertexPostprocessOperation vertex_postprocess_op)
{
    #ifdef __PRINT_DETAILED_STATS__
    #pragma omp barrier
    double t1 = omp_get_wtime();
    #pragma omp barrier
    #endif

    for(int front_pos = _first_vertex; front_pos < _last_vertex; front_pos++)
    {
        const int src_id = front_pos; //frontier_ids[front_pos];
        if(_frontier_flags[src_id] > 0)
        {
            const long long int start = _vertex_pointers[src_id];
            const long long int end = _vertex_pointers[src_id + 1];
            const int connections_count = end - start;

            vertex_preprocess_op(src_id, connections_count);

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

                edge_op(src_id, dst_id, local_edge_pos, global_edge_pos, vector_index);
            }

            vertex_postprocess_op(src_id, connections_count);
        }
    }

    #ifdef __PRINT_DETAILED_STATS__
    #pragma omp barrier
    double t2 = omp_get_wtime();
    #pragma omp master
    {
        cout << "first time: " << (t2 - t1)*1000 << " ms" << endl;
        cout << "first BW: " << (sizeof(int)*5.0)*(_vertex_pointers[_last_vertex] - _vertex_pointers[_first_vertex])/((t2-t1)*1e9) << " GB/s" << endl;
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
                                                       VertexPostprocessOperation vertex_postprocess_op)
{
    #ifdef __PRINT_DETAILED_STATS__
    #pragma omp barrier
    double t1 = omp_get_wtime();
    #pragma omp barrier
    #endif

    #pragma omp for schedule(static, 8)
    for (int front_pos = _first_vertex; front_pos < _last_vertex; front_pos++)
    {
        const int src_id = front_pos;//frontier_ids[front_pos];
        if(_frontier_flags[src_id] > 0)
        {
            const long long int start = _vertex_pointers[src_id];
            const long long int end = _vertex_pointers[src_id + 1];
            const int connections_count = end - start;

            vertex_preprocess_op(src_id, connections_count);

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

                    edge_op(src_id, dst_id, local_edge_pos, global_edge_pos, vector_index);
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

                edge_op(src_id, dst_id, local_edge_pos, global_edge_pos, vector_index);
            }

            vertex_postprocess_op(src_id, connections_count);
        }
    }

    #ifdef __PRINT_DETAILED_STATS__
    #pragma omp barrier
    double t2 = omp_get_wtime();
    #pragma omp master
    {
        cout << "second time: " << (t2 - t1)*1000 << " ms" << endl;
        cout << "second BW: " << (sizeof(int)*5.0)*(_vertex_pointers[_last_vertex] - _vertex_pointers[_first_vertex])/((t2-t1)*1e9) << " GB/s" << endl;
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
                                                             VertexPostprocessOperation vertex_postprocess_op)
{
    #ifdef __PRINT_DETAILED_STATS__
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

    #pragma omp for schedule(static, 1)
    for(int front_pos = _first_vertex; front_pos < _last_vertex; front_pos += VECTOR_LENGTH)
    {
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            if((front_pos + i) < _last_vertex)
            {
                int src_id = front_pos + i;//frontier_ids[front_pos + i];
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

        int max_connections = _vertex_pointers[front_pos + 1] - _vertex_pointers[front_pos];

        for(int edge_pos = 0; edge_pos < max_connections; edge_pos++)
        {
            #pragma _NEC ivdep
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                if(((front_pos + i) < _last_vertex) && (edge_pos < reg_connections[i]))
                {
                    const int src_id = front_pos + i;//frontier_ids[front_pos + i];
                    if(_frontier_flags[src_id] > 0)
                    {
                        const int vector_index = i;
                        const long long int global_edge_pos = reg_start[i] + edge_pos;
                        const int local_edge_pos = edge_pos;
                        const int dst_id = _adjacent_ids[global_edge_pos];

                        edge_op(src_id, dst_id, local_edge_pos, global_edge_pos, vector_index);
                    }
                }
            }
        }
    }

    #ifdef __PRINT_DETAILED_STATS__
    #pragma omp barrier
    double t2 = omp_get_wtime();
    #pragma omp master
    {
        cout << "third time: " << (t2 - t1)*1000 << " ms" << endl;
        cout << "third BW: " << (sizeof(int)*5.0)*(_vertex_pointers[ _last_vertex] - _vertex_pointers[_first_vertex])/((t2-t1)*1e9) << " GB/s" << endl << endl;
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
                                                                VertexPostprocessOperation vertex_postprocess_op)
{
    #ifdef __PRINT_DETAILED_STATS__
    #pragma omp barrier
    double t1 = omp_get_wtime();
    #pragma omp barrier
    #endif

    #pragma omp for schedule(static, 1)
    for(int cur_vector_segment = 0; cur_vector_segment < _ve_vector_segments_count; cur_vector_segment++)
    {
        int segment_first_vertex = cur_vector_segment * VECTOR_LENGTH + _ve_starting_vertex;

        long long segment_edges_start = _ve_vector_group_ptrs[cur_vector_segment];
        int segment_connections_count = _ve_vector_group_sizes[cur_vector_segment];

        for (int edge_pos = 0; edge_pos < segment_connections_count; edge_pos++)
        {
            #pragma _NEC ivdep
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            for (int i = 0; i < VECTOR_LENGTH; i++)
            {
                const int src_id = segment_first_vertex + i;

                //if(_frontier_flags[src_id] > 0)
                //{
                    const int vector_index = i;
                    const long long int global_edge_pos = segment_edges_start + edge_pos * VECTOR_LENGTH + i;
                    const int local_edge_pos = edge_pos;
                    const int dst_id = _ve_adjacent_ids[global_edge_pos];

                    edge_op(src_id, dst_id, local_edge_pos, global_edge_pos, vector_index);
                //}
            }
        }
    }

    #ifdef __PRINT_DETAILED_STATS__
    #pragma omp barrier
    double t2 = omp_get_wtime();
    #pragma omp master
    {
        double work = _ve_vector_group_ptrs[_ve_vector_segments_count - 1] - _ve_vector_group_ptrs[0];
        cout << "third time: " << (t2 - t1)*1000 << " ms" << endl;
        cout << "third BW: " << (sizeof(int)*5.0)*(work)/((t2-t1)*1e9) << " GB/s" << endl << endl;
    };
    #pragma omp barrier
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation, typename CollectiveEdgeOperation>
void GraphPrimitivesNEC::advance(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                 FrontierNEC &_frontier,
                                 EdgeOperation edge_op,
                                 VertexPreprocessOperation vertex_preprocess_op,
                                 VertexPostprocessOperation vertex_postprocess_op,
                                 CollectiveEdgeOperation collective_edge_op,
                                 bool _use_vector_extension)
{
    #pragma omp barrier

    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    const long long int *vertex_pointers = outgoing_ptrs;
    const int *adjacent_ids = outgoing_ids;
    const int *ve_adjacent_ids = ve_outgoing_ids;

    const int vector_engine_threshold_start = 0;
    const int vector_engine_threshold_end = _graph.get_nec_vector_engine_threshold_vertex();
    const int vector_core_threshold_start = _graph.get_nec_vector_engine_threshold_vertex();
    const int vector_core_threshold_end = _graph.get_nec_vector_core_threshold_vertex();
    const int collective_threshold_start = _graph.get_nec_vector_core_threshold_vertex();
    const int collective_threshold_end = _graph.get_vertices_count();

    int *frontier_flags = _frontier.frontier_flags;

    vector_engine_per_vertex_kernel(vertex_pointers, adjacent_ids, frontier_flags, vector_engine_threshold_start,
                                    vector_engine_threshold_end, edge_op, EMPTY_OP, EMPTY_OP);

    vector_core_per_vertex_kernel(vertex_pointers, adjacent_ids, frontier_flags, vector_core_threshold_start,
                                  vector_core_threshold_end, edge_op, EMPTY_OP, vertex_postprocess_op);

    if(_use_vector_extension == false) {
        collective_vertex_processing_kernel(vertex_pointers, adjacent_ids, frontier_flags, collective_threshold_start,
                                            collective_threshold_end, collective_edge_op, EMPTY_OP, EMPTY_OP);
    }
    else {
        ve_collective_vertex_processing_kernel(ve_vector_group_ptrs, ve_vector_group_sizes, ve_adjacent_ids,
                                               ve_vertices_count, ve_starting_vertex, ve_vector_segments_count,
                                               frontier_flags, collective_threshold_start, collective_threshold_end,
                                               collective_edge_op, EMPTY_OP, EMPTY_OP);
    }

    #pragma omp barrier
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
