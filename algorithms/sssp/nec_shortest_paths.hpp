#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ShortestPaths<_TVertexValue, _TEdgeWeight>::nec_dijkstra(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                              int _source_vertex,
                                                              _TEdgeWeight *_distances)
{
    LOAD_VECTORISED_CSR_GRAPH_REVERSE_DATA(_graph)

    /*int threads_count = omp_get_max_threads();
    _reversed_graph.set_threads_count(threads_count);

    #ifdef __USE_NEC_SX_AURORA__
    #pragma _NEC retain(_distances)
    #endif

    _reversed_graph.template vertex_array_set_to_constant<_TEdgeWeight>(_distances, FLT_MAX);
    _reversed_graph.template vertex_array_set_element<_TEdgeWeight>(_distances, _source_vertex, 0.0);

    _TEdgeWeight *cached_distances = _reversed_graph.template allocate_private_caches<_TEdgeWeight>(threads_count);

    double t1 = omp_get_wtime();
    int changes = 1;
    int iterations_count = 0;
    #pragma omp parallel num_threads(threads_count) shared(changes)
    {
        int reg_changes[VECTOR_LENGTH];
        _TEdgeWeight reg_distances[VECTOR_LENGTH];

        #ifdef __USE_NEC_SX_AURORA__
        #pragma _NEC vreg(reg_changes)
        #pragma _NEC vreg(reg_distances)
        #endif

        #ifdef __USE_NEC_SX_AURORA__
        #pragma _NEC vector
        #endif
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            reg_distances[i] = 0.0;
            reg_changes[i] = 0;
        }

        int thread_id = omp_get_thread_num();
        _TEdgeWeight *private_distances = &cached_distances[thread_id * CACHED_VERTICES * CACHE_STEP];

        while(changes > 0)
        {
            #pragma omp barrier

            _reversed_graph.template place_data_into_cache<_TEdgeWeight>(_distances, private_distances);

            #pragma omp barrier

            #ifdef __USE_NEC_SX_AURORA__
            #pragma _NEC vector
            #endif
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                reg_changes[i] = 0;
            }

            #pragma omp master
            {
                iterations_count++;
            }
            changes = 0;

            if(number_of_vertices_in_first_part > 0)
            {
                int local_changes = 0;
                for(int src_id = 0; src_id < number_of_vertices_in_first_part; src_id++)
                {
                    _TEdgeWeight shortest_distance = _distances[src_id];
                    long long edge_start = first_part_ptrs[src_id];
                    int connections_count = first_part_sizes[src_id];

                    #ifdef __USE_NEC_SX_AURORA__
                    #pragma _NEC vector
                    #endif
                    for(int i = 0; i < VECTOR_LENGTH; i++)
                    {
                        reg_distances[i] = shortest_distance;
                    }

                    #pragma omp for schedule(static, 1)
                    for(long long edge_pos = 0; edge_pos < connections_count; edge_pos += VECTOR_LENGTH)
                    {
                        #ifdef __USE_NEC_SX_AURORA__
                        #pragma _NEC ivdep
                        #pragma _NEC vovertake
                        #pragma _NEC novob
                        #pragma _NEC vector
                        #endif
                        for(int i = 0; i < VECTOR_LENGTH; i++)
                        {
                            int dst_id = incoming_ids[edge_start + edge_pos + i];
                            _TEdgeWeight weight = incoming_weights[edge_start + edge_pos + i];
                            _TEdgeWeight new_weight = weight + _reversed_graph.template load_vertex_data_cached<_TEdgeWeight>(dst_id, _distances, private_distances);

                            if(new_weight < reg_distances[i])
                                reg_distances[i] = new_weight;
                        }
                    }

                    shortest_distance = FLT_MAX;
                    #ifdef __USE_NEC_SX_AURORA__
                    #pragma _NEC vector
                    #endif
                    for(int i = 0; i < VECTOR_LENGTH; i++)
                    {
                        if(reg_distances[i] < shortest_distance)
                            shortest_distance = reg_distances[i];
                    }

                    #pragma omp critical
                    {
                        if(_distances[src_id] > shortest_distance)
                        {
                            _distances[src_id] = shortest_distance;
                            local_changes = 1;
                        }
                    }
                }

                #pragma omp atomic
                changes += local_changes;
            }

            #pragma omp for schedule(static, 1)
            for(int cur_vector_segment = 0; cur_vector_segment < vector_segments_count; cur_vector_segment++)
            {
                int segment_first_vertex = cur_vector_segment * VECTOR_LENGTH + number_of_vertices_in_first_part;

                long long segement_edges_start = vector_group_ptrs[cur_vector_segment];
                int segment_connections_count  = vector_group_sizes[cur_vector_segment];

                #ifdef __USE_NEC_SX_AURORA__
                #pragma _NEC vector
                #endif
                for(int i = 0; i < VECTOR_LENGTH; i++)
                {
                    int src_id = segment_first_vertex + i;
                    reg_distances[i] = _distances[src_id];
                }

                for(long long edge_pos = 0; edge_pos < segment_connections_count; edge_pos++)
                {
                    #ifdef __USE_NEC_SX_AURORA__
                    #pragma _NEC ivdep
                    #pragma _NEC vovertake
                    #pragma _NEC novob
                    #pragma _NEC vector
                    #endif
                    for(int i = 0; i < VECTOR_LENGTH; i++)
                    {
                        int src_id = segment_first_vertex + i;
                        int dst_id = incoming_ids[segement_edges_start + edge_pos * VECTOR_LENGTH + i];

                        _TEdgeWeight weight = incoming_weights[segement_edges_start + edge_pos * VECTOR_LENGTH + i];
                        _TEdgeWeight new_weight = weight +_reversed_graph.template load_vertex_data_cached<_TEdgeWeight>(dst_id, _distances, private_distances);

                        if(reg_distances[i] > new_weight)
                        {
                            reg_distances[i] = new_weight;
                            reg_changes[i] = 1;
                        }
                    }
                }

                #ifdef __USE_NEC_SX_AURORA__
                #pragma _NEC vector
                #endif
                for(int i = 0; i < VECTOR_LENGTH; i++)
                {
                    int src_id = segment_first_vertex + i;
                    _distances[src_id] = reg_distances[i];
                }
            }

            #pragma omp barrier

            int private_changes = 0;
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                private_changes += reg_changes[i];
            }

            #pragma omp atomic
            changes += private_changes;

            #pragma omp barrier
        }
    }
    double t2 = omp_get_wtime();

    #ifdef __PRINT_DETAILED_STATS__
    print_performance_stats(edges_count, iterations_count, t2 - t1);
    #endif

    _reversed_graph.template free_data<_TEdgeWeight>(cached_distances);*/
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
