#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
void GraphAbstractionsMulticore::vector_engine_per_vertex_kernel_all_active(VectorCSRGraph &_graph,
                                                                      const int _first_vertex,
                                                                      const int _last_vertex,
                                                                      EdgeOperation edge_op,
                                                                      VertexPreprocessOperation vertex_preprocess_op,
                                                                      VertexPostprocessOperation vertex_postprocess_op,
                                                                      const int _first_edge,
                                                                      const long long _shard_shift,
                                                                      bool _outgoing_graph_is_stored)
{
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    Timer tm;
    tm.start();
    #endif

    LOAD_VECTOR_CSR_GRAPH_DATA(_graph);

    TraversalDirection traversal = current_traversal_direction;
    int storage = CSR_STORAGE;
    long long process_shift = compute_process_shift(_shard_shift, traversal, storage, edges_count,
                                                    _outgoing_graph_is_stored);

    for(int front_pos = _first_vertex; front_pos < _last_vertex; front_pos++)
    {
        const int src_id = front_pos;

        const long long int start = vertex_pointers[src_id];
        const long long int end = vertex_pointers[src_id + 1];
        const int connections_count = end - start;

        vertex_preprocess_op(src_id, connections_count, 0);

        #pragma omp for schedule(static, 1)
        for(int vec_start = 0; vec_start < connections_count; vec_start += VECTOR_LENGTH)
        {
            #pragma simd
            #pragma vector
            #pragma ivdep
            #pragma unroll(VECTOR_LENGTH)
            for (int i = 0; i < VECTOR_LENGTH; i++)
            {
                int local_edge_pos = vec_start + i;
                const long long internal_edge_pos = start + local_edge_pos;
                const int vector_index = i;
                const long long external_edge_pos = process_shift + internal_edge_pos;

                if (local_edge_pos < connections_count)
                {
                    const int dst_id = adjacent_ids[internal_edge_pos];
                    edge_op(src_id, dst_id, local_edge_pos, external_edge_pos, vector_index);
                }
            }
        }

        vertex_postprocess_op(src_id, connections_count, 0);
    }

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.end();
    long long work = vertex_pointers[_last_vertex] - vertex_pointers[_first_vertex];
    tm.print_time_and_bandwidth_stats("Advance (all, ve)", work, INT_ELEMENTS_PER_EDGE*sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
void GraphAbstractionsMulticore::vector_core_per_vertex_kernel_all_active(VectorCSRGraph &_graph,
                                                                    const int _first_vertex,
                                                                    const int _last_vertex,
                                                                    EdgeOperation edge_op,
                                                                    VertexPreprocessOperation vertex_preprocess_op,
                                                                    VertexPostprocessOperation vertex_postprocess_op,
                                                                    const int _first_edge,
                                                                    const long long _shard_shift,
                                                                    bool _outgoing_graph_is_stored)
{
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    Timer tm;
    tm.start();
    #endif

    LOAD_VECTOR_CSR_GRAPH_DATA(_graph);

    TraversalDirection traversal = current_traversal_direction;
    int storage = CSR_STORAGE;
    long long process_shift = compute_process_shift(_shard_shift, traversal, storage, edges_count,
                                                    _outgoing_graph_is_stored);

    #pragma omp for schedule(static, 1)
    for (int src_id = _first_vertex; src_id < _last_vertex; src_id++)
    {
        const long long int start = vertex_pointers[src_id];
        const long long int end = vertex_pointers[src_id + 1];
        const int connections_count = end - start;

        vertex_preprocess_op(src_id, connections_count, 0);

        for (int vec_start = 0; vec_start < connections_count; vec_start += VECTOR_LENGTH)
        {
            #pragma simd
            #pragma vector
            #pragma ivdep
            #pragma unroll(VECTOR_LENGTH)
            for (int i = 0; i < VECTOR_LENGTH; i++)
            {
                int local_edge_pos = vec_start + i;
                const long long internal_edge_pos = start + local_edge_pos;
                const long long int global_edge_pos = start + local_edge_pos;
                const int vector_index = i;
                const long long external_edge_pos = process_shift + internal_edge_pos;

                if (local_edge_pos < connections_count)
                {
                    const int dst_id = adjacent_ids[internal_edge_pos];
                    edge_op(src_id, dst_id, local_edge_pos, external_edge_pos, vector_index);
                }
            }
        }

        vertex_postprocess_op(src_id, connections_count, 0);
    }

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.end();
    long long work = vertex_pointers[_last_vertex] - vertex_pointers[_first_vertex];
    tm.print_time_and_bandwidth_stats("Advance (all, vc)", work, INT_ELEMENTS_PER_EDGE*sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
void GraphAbstractionsMulticore::ve_collective_vertex_processing_kernel_all_active(VectorCSRGraph &_graph,
                                                                             const int _first_vertex,
                                                                             const int _last_vertex,
                                                                             EdgeOperation edge_op,
                                                                             VertexPreprocessOperation vertex_preprocess_op,
                                                                             VertexPostprocessOperation vertex_postprocess_op,
                                                                             const int _first_edge,
                                                                             const long long _shard_shift,
                                                                             bool _outgoing_graph_is_stored)
{
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    Timer tm;
    tm.start();
    #endif

    LOAD_VECTOR_CSR_GRAPH_DATA(_graph);

    TraversalDirection traversal = current_traversal_direction;
    int storage = VE_STORAGE;
    long long process_shift = compute_process_shift(_shard_shift, traversal, storage, edges_count,
                                                    _outgoing_graph_is_stored);

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
    for(int cur_vector_segment = 0; cur_vector_segment < ve_vector_segments_count; cur_vector_segment++)
    {
        int segment_first_vertex = cur_vector_segment * VECTOR_LENGTH + ve_starting_vertex;

        long long segment_edges_start = ve_vector_group_ptrs[cur_vector_segment];
        int segment_connections_count = ve_vector_group_sizes[cur_vector_segment];

        #pragma simd
        #pragma vector
        #pragma ivdep
        #pragma unroll(VECTOR_LENGTH)
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            int src_id = segment_first_vertex + i;
            reg_real_start[i] = vertex_pointers[src_id];

            if(segment_connections_count > 0)
                reg_real_connections_count[i] = vertex_pointers[src_id + 1] - reg_real_start[i];
            else
                reg_real_connections_count[i] = 0;

            if(src_id < vertices_count)
                vertex_preprocess_op(src_id, reg_real_connections_count[i], i);
        }

        for(int edge_pos = _first_edge; edge_pos < segment_connections_count; edge_pos++)
        {
            #pragma simd
            #pragma vector
            #pragma ivdep
            #pragma unroll(VECTOR_LENGTH)
            for (int i = 0; i < VECTOR_LENGTH; i++)
            {
                const int src_id = segment_first_vertex + i;

                const int vector_index = i;
                const long long internal_edge_pos = segment_edges_start + edge_pos * VECTOR_LENGTH + i;
                const int local_edge_pos = edge_pos;
                const long long external_edge_pos = process_shift + internal_edge_pos;

                if((src_id < vertices_count) && (edge_pos < reg_real_connections_count[i]))
                {
                    const int dst_id = ve_adjacent_ids[internal_edge_pos];
                    edge_op(src_id, dst_id, local_edge_pos, external_edge_pos, vector_index);
                }
            }
        }

        #pragma simd
        #pragma vector
        #pragma ivdep
        #pragma unroll(VECTOR_LENGTH)
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            int src_id = segment_first_vertex + i;

            if(src_id < vertices_count)
                vertex_postprocess_op(src_id, reg_real_connections_count[i], i);
        }
    }

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.end();
    long long work = _graph.get_edges_count_in_ve();
    tm.print_time_and_bandwidth_stats("Advance (all, collective)", work, INT_ELEMENTS_PER_EDGE*sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
