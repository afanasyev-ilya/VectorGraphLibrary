#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
void GraphAbstractionsMulticore::vector_engine_per_vertex_kernel_sparse(UndirectedCSRGraph &_graph,
                                                                  FrontierMulticore &_frontier,
                                                                  EdgeOperation edge_op,
                                                                  VertexPreprocessOperation vertex_preprocess_op,
                                                                  VertexPostprocessOperation vertex_postprocess_op,
                                                                  const int _first_edge,
                                                                  bool _outgoing_graph_is_stored)
{
    Timer tm;
    tm.start();

    LOAD_UNDIRECTED_CSR_GRAPH_DATA(_graph);
    int *frontier_ids = &(_frontier.get_ids()[0]);
    int frontier_segment_size = _frontier.get_vector_engine_part_size();

    TraversalDirection traversal = current_traversal_direction;
    int storage = CSR_STORAGE;
    long long process_shift = compute_process_shift(0/*shard shift*/, traversal, storage, edges_count, _outgoing_graph_is_stored);

    DelayedWriteNEC delayed_write;
    delayed_write.init();

    for (int front_pos = 0; front_pos < frontier_segment_size; front_pos++)
    {
        const int src_id = frontier_ids[front_pos];

        const long long int start = vertex_pointers[src_id];
        const long long int end = vertex_pointers[src_id + 1];
        const int connections_count = end - start;

        vertex_preprocess_op(src_id, connections_count, 0, delayed_write);

        #pragma omp for schedule(static, 8)
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
                const int vector_index = i;
                const long long external_edge_pos = process_shift + internal_edge_pos;

                if (local_edge_pos < connections_count)
                {
                    const int dst_id = adjacent_ids[internal_edge_pos];
                    edge_op(src_id, dst_id, local_edge_pos, external_edge_pos, vector_index, delayed_write);
                }
            }
        }

        vertex_postprocess_op(src_id, connections_count, 0, delayed_write);
    }

    tm.end();
    long long work = _frontier.get_vector_engine_part_neighbours_count();
    performance_stats.update_advance_ve_part_time(tm);
    performance_stats.update_graph_processing_stats(work*INT_ELEMENTS_PER_EDGE*sizeof(int), work);
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_time_and_bandwidth_stats("Advance (sparse, ve)", work, INT_ELEMENTS_PER_EDGE*sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
void GraphAbstractionsMulticore::vector_core_per_vertex_kernel_sparse(UndirectedCSRGraph &_graph,
                                                                FrontierMulticore &_frontier,
                                                                EdgeOperation edge_op,
                                                                VertexPreprocessOperation vertex_preprocess_op,
                                                                VertexPostprocessOperation vertex_postprocess_op,
                                                                const int _first_edge,
                                                                bool _outgoing_graph_is_stored)
{
    Timer tm;
    tm.start();

    LOAD_UNDIRECTED_CSR_GRAPH_DATA(_graph);
    int *frontier_ids = &(_frontier.get_ids()[_frontier.get_vector_engine_part_size()]);
    int frontier_segment_size = _frontier.get_vector_core_part_size();

    TraversalDirection traversal = current_traversal_direction;
    int storage = CSR_STORAGE;
    long long process_shift = compute_process_shift(0/*shard shift*/, traversal, storage, edges_count, _outgoing_graph_is_stored);

    DelayedWriteNEC delayed_write;
    delayed_write.init();

    #pragma omp for schedule(static, 2)
    for (int front_pos = 0; front_pos < frontier_segment_size; front_pos++)
    {
        const int src_id = frontier_ids[front_pos];

        const long long int start = vertex_pointers[src_id];
        const long long int end = vertex_pointers[src_id + 1];
        const int connections_count = end - start;

        vertex_preprocess_op(src_id, connections_count, 0, delayed_write);

        #pragma simd
        #pragma vector
        #pragma ivdep
        #pragma unroll(VECTOR_LENGTH)
        for (int local_edge_pos = 0; local_edge_pos < connections_count; local_edge_pos++)
        {
            const long long internal_edge_pos = start + local_edge_pos;
            const int vector_index = get_vector_index(local_edge_pos);
            const int dst_id = adjacent_ids[internal_edge_pos];
            const long long external_edge_pos = process_shift + internal_edge_pos;

            edge_op(src_id, dst_id, local_edge_pos, external_edge_pos, vector_index, delayed_write);
        }

        vertex_postprocess_op(src_id, connections_count, 0, delayed_write);
    }

    tm.end();
    long long work = _frontier.get_vector_core_part_neighbours_count();
    performance_stats.update_advance_vc_part_time(tm);
    performance_stats.update_graph_processing_stats(work*INT_ELEMENTS_PER_EDGE*sizeof(int), work);
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_time_and_bandwidth_stats("Advance (sparse, vc)", work, INT_ELEMENTS_PER_EDGE*sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
void GraphAbstractionsMulticore::collective_vertex_processing_kernel_sparse(UndirectedCSRGraph &_graph,
                                                                      FrontierMulticore &_frontier,
                                                                      const int _first_vertex,
                                                                      const int _last_vertex,
                                                                      EdgeOperation edge_op,
                                                                      VertexPreprocessOperation vertex_preprocess_op,
                                                                      VertexPostprocessOperation vertex_postprocess_op,
                                                                      const int _first_edge,
                                                                      bool _outgoing_graph_is_stored)
{
    Timer tm;
    tm.start();

    LOAD_UNDIRECTED_CSR_GRAPH_DATA(_graph);
    int *frontier_ids = &(_frontier.get_ids()[_frontier.get_vector_core_part_size() + _frontier.get_vector_engine_part_size()]);
    int frontier_segment_size = _frontier.get_collective_part_size();

    TraversalDirection traversal = current_traversal_direction;
    int storage = CSR_STORAGE;
    long long process_shift = compute_process_shift(0/*shard shift*/, traversal, storage, edges_count, _outgoing_graph_is_stored);

    long long int reg_start[VECTOR_LENGTH];
    long long int reg_end[VECTOR_LENGTH];
    int reg_connections[VECTOR_LENGTH];

    #pragma _NEC vreg(reg_start)
    #pragma _NEC vreg(reg_end)
    #pragma _NEC vreg(reg_connections)

    #pragma simd
    #pragma vector
    #pragma ivdep
    #pragma unroll(VECTOR_LENGTH)
    for(int i = 0; i < VECTOR_LENGTH; i++)
    {
        reg_start[i] = 0;
        reg_end[i] = 0;
        reg_connections[i] = 0;
    }

    DelayedWriteNEC delayed_write;
    delayed_write.init();

    #pragma omp for schedule(static, 4)
    for(int front_pos = 0; front_pos < frontier_segment_size; front_pos += VECTOR_LENGTH)
    {
        #pragma simd
        #pragma vector
        #pragma ivdep
        #pragma unroll(VECTOR_LENGTH)
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            if((front_pos + i) < frontier_segment_size)
            {
                int src_id = frontier_ids[front_pos + i];
                reg_start[i] = vertex_pointers[src_id];
                reg_end[i] = vertex_pointers[src_id + 1];
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
        #pragma simd
        #pragma vector
        #pragma ivdep
        #pragma unroll(VECTOR_LENGTH)
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
                #pragma simd
                #pragma vector
                #pragma ivdep
                #pragma unroll(VECTOR_LENGTH)
                for (int i = 0; i < VECTOR_LENGTH; i++)
                {
                    if (((front_pos + i) < frontier_segment_size) && (edge_pos < reg_connections[i]))
                    {
                        const int src_id = frontier_ids[front_pos + i];
                        const int vector_index = i;
                        const long long int internal_edge_pos = reg_start[i] + edge_pos;
                        const int local_edge_pos = edge_pos;
                        const int dst_id = adjacent_ids[internal_edge_pos];
                        const long long external_edge_pos = process_shift + internal_edge_pos;

                        edge_op(src_id, dst_id, local_edge_pos, external_edge_pos, vector_index, delayed_write);
                    }
                }
            }

            #pragma simd
            #pragma vector
            #pragma ivdep
            #pragma unroll(VECTOR_LENGTH)
            for (int i = 0; i < VECTOR_LENGTH; i++)
            {
                if ((front_pos + i) < frontier_segment_size)
                {
                    int src_id = frontier_ids[front_pos + i];
                    vertex_postprocess_op(src_id, reg_connections[i], i, delayed_write);
                }
            }
        }
    }

    tm.end();
    long long work = _frontier.get_collective_part_neighbours_count();
    performance_stats.update_advance_collective_part_time(tm);
    performance_stats.update_graph_processing_stats(work*INT_ELEMENTS_PER_EDGE*sizeof(int), work);
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_time_and_bandwidth_stats("Advance (sparse, collective)", work, INT_ELEMENTS_PER_EDGE*sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
