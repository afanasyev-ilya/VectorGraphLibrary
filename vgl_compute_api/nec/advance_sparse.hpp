#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
void GraphAbstractionsNEC::vector_engine_per_vertex_kernel_sparse(VectorCSRGraph &_graph,
                                                                  FrontierVectorCSR &_frontier,
                                                                  EdgeOperation edge_op,
                                                                  VertexPreprocessOperation vertex_preprocess_op,
                                                                  VertexPostprocessOperation vertex_postprocess_op)
{
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    Timer tm;
    tm.start();
    #endif

    long long *vertex_pointers = _graph.get_vertex_pointers ();
    int *adjacent_ids          = _graph.get_adjacent_ids    ();
    long long int edges_count  = _graph.get_edges_count     ();
    int *frontier_ids          = &(_frontier.get_ids()[0]);
    int frontier_segment_size  = _frontier.get_vector_engine_part_size();

    TraversalDirection traversal = current_traversal_direction;
    int storage = CSR_STORAGE;
    long long process_shift = compute_process_shift(traversal, storage);

    for (int front_pos = 0; front_pos < frontier_segment_size; front_pos++)
    {
        const int src_id = frontier_ids[front_pos];

        const long long int start = vertex_pointers[src_id];
        const long long int end = vertex_pointers[src_id + 1];
        const int connections_count = end - start;

        vertex_preprocess_op(src_id, connections_count, 0);

        if(use_safe_stores) // all vector stores are safe in this branch (vob)
        {
            #pragma _NEC novector
            #pragma omp for schedule(static, 8)
            for (int vec_start = 0; vec_start < connections_count; vec_start += VECTOR_LENGTH)
            {
                #pragma _NEC cncall
                #pragma _NEC ivdep
                #pragma _NEC vector
                #pragma _NEC gather_reorder
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
        }
        else
        {
            #pragma _NEC novector
            #pragma omp for schedule(static, 8)
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
                    const int vector_index = i;
                    const long long external_edge_pos = process_shift + internal_edge_pos;

                    if (local_edge_pos < connections_count)
                    {
                        const int dst_id = adjacent_ids[internal_edge_pos];
                        edge_op(src_id, dst_id, local_edge_pos, external_edge_pos, vector_index);
                    }
                }
            }
        }

        vertex_postprocess_op(src_id, connections_count, 0);
    }

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.end();
    long long work = _frontier.get_vector_engine_part_neighbours_count();
    tm.print_time_and_bandwidth_stats("Advance (sparse, ve)", work, INT_ELEMENTS_PER_EDGE*sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
void GraphAbstractionsNEC::vector_core_per_vertex_kernel_sparse(VectorCSRGraph &_graph,
                                                                FrontierVectorCSR &_frontier,
                                                                EdgeOperation edge_op,
                                                                VertexPreprocessOperation vertex_preprocess_op,
                                                                VertexPostprocessOperation vertex_postprocess_op)
{
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    Timer tm;
    tm.start();
    #endif

    long long *vertex_pointers = _graph.get_vertex_pointers ();
    int *adjacent_ids          = _graph.get_adjacent_ids    ();
    long long int edges_count  = _graph.get_edges_count     ();
    int *frontier_ids          = &(_frontier.get_ids()[_frontier.get_vector_engine_part_size()]);
    int frontier_segment_size  = _frontier.get_vector_core_part_size();

    TraversalDirection traversal = current_traversal_direction;
    int storage = CSR_STORAGE;
    long long process_shift = compute_process_shift(traversal, storage);

    #pragma omp for schedule(static)
    for (int front_pos = 0; front_pos < frontier_segment_size; front_pos++)
    {
        const int src_id = frontier_ids[front_pos];

        const long long int start = vertex_pointers[src_id];
        const long long int end = vertex_pointers[src_id + 1];
        const int connections_count = end - start;

        vertex_preprocess_op(src_id, connections_count, 0);

        if(use_safe_stores) // all vector stores are safe in this branch (vob)
        {
            #pragma _NEC cncall
            #pragma _NEC ivdep
            #pragma _NEC vob
            #pragma _NEC vector
            #pragma _NEC gather_reorder
            for (int local_edge_pos = 0; local_edge_pos < connections_count; local_edge_pos++)
            {
                const long long internal_edge_pos = start + local_edge_pos;
                const int vector_index = get_vector_index(local_edge_pos);
                const int dst_id = adjacent_ids[internal_edge_pos];
                const long long external_edge_pos = process_shift + internal_edge_pos;

                edge_op(src_id, dst_id, local_edge_pos, external_edge_pos, vector_index);
            }
        }
        else
        {
            #pragma _NEC cncall
            #pragma _NEC ivdep
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vob
            #pragma _NEC vector
            #pragma _NEC gather_reorder
            for (int local_edge_pos = 0; local_edge_pos < connections_count; local_edge_pos++)
            {
                const long long internal_edge_pos = start + local_edge_pos;
                const int vector_index = get_vector_index(local_edge_pos);
                const int dst_id = adjacent_ids[internal_edge_pos];
                const long long external_edge_pos = process_shift + internal_edge_pos;

                edge_op(src_id, dst_id, local_edge_pos, external_edge_pos, vector_index);
            }
        }

        vertex_postprocess_op(src_id, connections_count, 0);
    }

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.end();
    long long work = _frontier.get_vector_core_part_neighbours_count();
    tm.print_time_and_bandwidth_stats("Advance (sparse, vc)", work, INT_ELEMENTS_PER_EDGE*sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
void GraphAbstractionsNEC::collective_vertex_processing_kernel_sparse(VectorCSRGraph &_graph,
                                                                      FrontierVectorCSR &_frontier,
                                                                      const int _first_vertex,
                                                                      const int _last_vertex,
                                                                      EdgeOperation edge_op,
                                                                      VertexPreprocessOperation vertex_preprocess_op,
                                                                      VertexPostprocessOperation vertex_postprocess_op)
{
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    Timer tm;
    tm.start();
    #endif

    long long *vertex_pointers = _graph.get_vertex_pointers ();
    int *adjacent_ids          = _graph.get_adjacent_ids    ();
    long long int edges_count  = _graph.get_edges_count     ();
    int vertices_count         = _graph.get_vertices_count  ();
    int *frontier_ids          = &(_frontier.get_ids()[_frontier.get_vector_core_part_size()
                                                       + _frontier.get_vector_engine_part_size()]);
    int frontier_segment_size  = _frontier.get_collective_part_size();

    TraversalDirection traversal = current_traversal_direction;
    int storage = CSR_STORAGE;
    long long process_shift = compute_process_shift(traversal, storage);

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

    #pragma omp for schedule(static, 4)
    for(int front_pos = 0; front_pos < frontier_segment_size; front_pos += VECTOR_LENGTH)
    {
        #pragma _NEC vector
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            if((front_pos + i) < frontier_segment_size)
            {
                int src_id = frontier_ids[front_pos + i];
                reg_start[i] = vertex_pointers[src_id];
                reg_end[i] = vertex_pointers[src_id + 1];
                reg_connections[i] = reg_end[i] - reg_start[i];
                vertex_preprocess_op(src_id, reg_connections[i], i);
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
            if(use_safe_stores) // all vector stores are safe in this branch (vob)
            {
                for (int edge_pos = 0; edge_pos < max_connections; edge_pos++)
                {
                    #pragma _NEC cncall
                    #pragma _NEC ivdep
                    #pragma _NEC vector
                    #pragma _NEC sparse
                    #pragma _NEC gather_reorder
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

                            edge_op(src_id, dst_id, local_edge_pos, external_edge_pos, vector_index);
                        }
                    }
                }
            }
            else // all vector stores are NOT safe in this branch (novob)
            {
                for (int edge_pos = 0; edge_pos < max_connections; edge_pos++)
                {
                    #pragma _NEC cncall
                    #pragma _NEC ivdep
                    #pragma _NEC vovertake
                    #pragma _NEC novob
                    #pragma _NEC vector
                    #pragma _NEC sparse
                    #pragma _NEC gather_reorder
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

                            edge_op(src_id, dst_id, local_edge_pos, external_edge_pos, vector_index);
                        }
                    }
                }
            }

            #pragma _NEC vector
            for (int i = 0; i < VECTOR_LENGTH; i++)
            {
                if ((front_pos + i) < frontier_segment_size)
                {
                    int src_id = frontier_ids[front_pos + i];
                    vertex_postprocess_op(src_id, reg_connections[i], i);
                }
            }
        }
    }

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.end();
    long long work = _frontier.get_collective_part_neighbours_count();
    tm.print_time_and_bandwidth_stats("Advance (sparse, collective)", work, INT_ELEMENTS_PER_EDGE*sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
