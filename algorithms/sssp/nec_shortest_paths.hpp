#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _TVertexValue, typename _TEdgeWeight>
void SSSP::nec_dijkstra_partial_active(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                       _TEdgeWeight *_distances,
                                       int _source_vertex)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);
    FrontierNEC all_active_frontier(vertices_count);

    int *was_changes;
    MemoryAPI::allocate_array(&was_changes, vertices_count);

    auto init_distances = [_distances, _source_vertex] (int src_id, int connections_count, int vector_index)
    {
        if(src_id == _source_vertex)
            _distances[_source_vertex] = 0;
        else
            _distances[src_id] = FLT_MAX;
    };

    auto init_changes = [was_changes, _source_vertex] (int src_id, int connections_count, int vector_index)
    {
        if(src_id == _source_vertex)
            was_changes[_source_vertex] = 1;
        else
            was_changes[src_id] = 0;
    };

    auto changes_occurred = [&was_changes] (int src_id)->int
    {
        int res = NOT_IN_FRONTIER_FLAG;
        if(was_changes[src_id] > 0)
            res = IN_FRONTIER_FLAG;
        return res;
    };

    graph_API.compute(_graph, all_active_frontier, init_distances); // init distances with all-active frontier
    graph_API.compute(_graph, all_active_frontier, init_changes); // init changes with all-active frontier

    graph_API.generate_new_frontier(_graph, frontier, changes_occurred); // reduce frontier to 1 source-vertex element

    double t1 = omp_get_wtime();
    int iterations_count = 0;
    while(frontier.size() > 0)
    {
        float *collective_outgoing_weights = graph_API.get_collective_weights(_graph, frontier);
        auto reset_changes = [was_changes] (int src_id, int connections_count, int vector_index)
        {
            was_changes[src_id] = 0;
        };
        graph_API.compute(_graph, all_active_frontier, reset_changes);

        #pragma omp parallel
        {
            NEC_REGISTER_INT(was_changes, 0);

            auto edge_op_push = [outgoing_weights, _distances, was_changes]
               (int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos,
                int vector_index, DelayedWriteNEC &delayed_write)
            {
                float weight = outgoing_weights[global_edge_pos];
                float dst_weight = _distances[dst_id];
                float src_weight = _distances[src_id];
                if(dst_weight > src_weight + weight)
                {
                    _distances[dst_id] = src_weight + weight;
                    was_changes[dst_id] = 1;
                    delayed_write.start_write(was_changes, 1, vector_index);
                }
            };

            auto edge_op_collective_push = [collective_outgoing_weights, _distances, was_changes]
                    (int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos,
                            int vector_index, DelayedWriteNEC &delayed_write)
            {
                float weight = collective_outgoing_weights[global_edge_pos];
                float dst_weight = _distances[dst_id];
                float src_weight = _distances[src_id];
                if(dst_weight > src_weight + weight)
                {
                    _distances[dst_id] = src_weight + weight;
                    was_changes[dst_id] = 1;
                    was_changes[src_id] = 1;
                }
            };

            struct VertexPostprocessFunctor
            {
                int *was_changes;
                int *reg_was_changes;
                VertexPostprocessFunctor(int *_was_changes, int *_reg_was_changes): was_changes(_was_changes), reg_was_changes(_reg_was_changes) {}
                void operator()(int src_id, int connections_count, int vector_index, DelayedWriteNEC &delayed_write)
                {
                    delayed_write.finish_write_max(was_changes, src_id);
                }
            };
            VertexPostprocessFunctor vertex_postprocess_op(was_changes, reg_was_changes);

            graph_API.advance(_graph, frontier, edge_op_push, EMPTY_VERTEX_OP, vertex_postprocess_op,
                               edge_op_collective_push, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);
        }

        graph_API.generate_new_frontier(_graph, frontier, changes_occurred);
        iterations_count++;
    }
    double t2 = omp_get_wtime();
    performance = edges_count / ((t2 - t1)*1e6);

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    PerformanceStats::print_performance_stats("partial active sssp (dijkstra)", t2 - t1, edges_count, iterations_count);
    #endif

    MemoryAPI::free_array(was_changes);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _TVertexValue, typename _TEdgeWeight>
void SSSP::nec_dijkstra_all_active(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                   _TEdgeWeight *_distances,
                                   int _source_vertex,
                                   TraversalDirection _traversal_direction)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);
    _TEdgeWeight *old_distances = class_old_distances;

    frontier.set_all_active();

    double t1 = omp_get_wtime();
    auto init_distances = [_distances, _source_vertex] (int src_id, int connections_count, int vector_index)
    {
        if(src_id == _source_vertex)
            _distances[_source_vertex] = 0;
        else
            _distances[src_id] = FLT_MAX;
    };
    graph_API.compute(_graph, frontier, init_distances);

    int iterations_count = 0;
    int changes = 1;
    while(changes)
    {
        changes = 0;
        float *collective_outgoing_weights = graph_API.get_collective_weights(_graph, frontier);

        auto save_old_distances = [_distances, old_distances] (int src_id, int connections_count, int vector_index)
        {
            old_distances[src_id] = _distances[src_id];
        };
        graph_API.compute(_graph, frontier, save_old_distances);

        #pragma omp parallel shared(changes)
        {
            NEC_REGISTER_FLT(distances, 0);

            if(_traversal_direction == PUSH_TRAVERSAL) // PUSH PART
            {
                auto edge_op_push = [outgoing_weights, _distances](int src_id, int dst_id, int local_edge_pos,
                    long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
                {
                    float weight = outgoing_weights[global_edge_pos];
                    float dst_weight = _distances[dst_id];
                    float src_weight = _distances[src_id];
                    if(dst_weight > src_weight + weight)
                    {
                        _distances[dst_id] = src_weight + weight;
                    }
                };

                auto edge_op_collective_push = [collective_outgoing_weights, _distances]
                        (int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos,
                                int vector_index, DelayedWriteNEC &delayed_write)
                {
                    float weight = collective_outgoing_weights[global_edge_pos];
                    float dst_weight = _distances[dst_id];
                    float src_weight = _distances[src_id];
                    if(dst_weight > src_weight + weight)
                    {
                        _distances[dst_id] = src_weight + weight;
                    }
                };

                graph_API.advance(_graph, frontier, edge_op_push, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP,
                                  edge_op_collective_push, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);
            }

            if(_traversal_direction == PULL_TRAVERSAL) // PULL PART
            {
                auto edge_op_pull = [outgoing_weights, _distances](int src_id, int dst_id, int local_edge_pos,
                    long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
                {
                    float weight = outgoing_weights[global_edge_pos];
                    float dst_weight = _distances[dst_id];
                    if(_distances[src_id] > dst_weight + weight)
                    {
                        _distances[src_id] = dst_weight + weight;
                    }
                };

                auto edge_op_collective_pull = [collective_outgoing_weights, _distances, &reg_distances]
                        (int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos,
                                int vector_index, DelayedWriteNEC &delayed_write)
                {
                    float weight = collective_outgoing_weights[global_edge_pos];
                    float dst_weight = _distances[dst_id];
                    if(reg_distances[vector_index] > dst_weight + weight)
                    {
                        reg_distances[vector_index] = dst_weight + weight;
                    }
                };

                auto vertex_preprocess_op_collective_pull = [_distances, &reg_distances]
                        (int src_id, int connections_count, int vector_index, DelayedWriteNEC &delayed_write)
                {
                    reg_distances[vector_index] = _distances[src_id];
                };

                auto vertex_postprocess_op_collective_pull = [_distances, &reg_distances]
                        (int src_id, int connections_count, int vector_index, DelayedWriteNEC &delayed_write)
                {
                    if(_distances[src_id] > reg_distances[vector_index])
                    {
                         _distances[src_id] = reg_distances[vector_index];
                    }
                };

                graph_API.advance(_graph, frontier, edge_op_pull, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP,
                                   edge_op_collective_pull, vertex_preprocess_op_collective_pull, vertex_postprocess_op_collective_pull);
            }
        }

        auto calculate_changes_count = [_distances, old_distances] (int src_id, int connections_count, int vector_index)->int
        {
            int result = 0;
            if(old_distances[src_id] != _distances[src_id])
                result = 1;
            return result;
        };
        changes = graph_API.reduce<int>(_graph, frontier, calculate_changes_count, REDUCE_SUM);

        iterations_count++;
    }
    double t2 = omp_get_wtime();
    performance = edges_count / ((t2 - t1)*1e6);

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    PerformanceStats::print_performance_stats("all active sssp (dijkstra)", t2 - t1, edges_count, iterations_count);
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _TVertexValue, typename _TEdgeWeight>
void SSSP::nec_dijkstra(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                        _TEdgeWeight *_distances,
                        int _source_vertex,
                        AlgorithmFrontierType _frontier_type,
                        TraversalDirection _traversal_direction)
{
    if(_frontier_type == PARTIAL_ACTIVE)
        nec_dijkstra_partial_active(_graph, _distances, _source_vertex);
    else if(_frontier_type == ALL_ACTIVE)
        nec_dijkstra_all_active(_graph, _distances, _source_vertex, _traversal_direction);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _TVertexValue, typename _TEdgeWeight>
void SSSP::nec_bellamn_ford(EdgesListGraph<_TVertexValue, _TEdgeWeight> &_graph,
                            _TEdgeWeight *_distances,
                            int _source_vertex)
{
    LOAD_EDGES_LIST_GRAPH_DATA(_graph);

    double t1 = omp_get_wtime();
    #pragma omp parallel for
    for(int i = 0; i < vertices_count; i++)
    {
        _distances[i] = FLT_MAX;
    }
    _distances[_source_vertex] = 0;

    int iterations_count = 0;
    int changes_count = 0;
    do
    {
        NEC_REGISTER_INT(changes, 0);

        auto edge_op = [_distances, weights, &reg_changes](int src_id, int dst_id, long long int global_edge_pos, int vector_index)
        {
            float weight = weights[global_edge_pos];
            float dst_weight = _distances[dst_id];
            float src_weight = _distances[src_id];
            if(dst_weight > src_weight + weight)
            {
                _distances[dst_id] = src_weight + weight;
                reg_changes[vector_index] = 1;
            }
        };

        graph_API.advance(_graph, edge_op);

        changes_count = register_sum_reduce(reg_changes);
        iterations_count++;
    } while(changes_count > 0);

    double t2 = omp_get_wtime();
    performance = edges_count / ((t2 - t1)*1e6);

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    PerformanceStats::print_performance_stats("sssp (bellman-ford)", t2 - t1, edges_count, iterations_count);
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

