#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
template <typename _T>
void SSSP::nec_dijkstra_partial_active(VectCSRGraph &_graph,
                                       EdgesArray_Vect<_T> &_weights,
                                       VerticesArray<_T> &_distances,
                                       int _source_vertex)
{
    #if defined(__USE_NEC_SX_AURORA__)
    GraphAbstractionsNEC graph_API(_graph);
    FrontierNEC work_frontier(_graph);
    FrontierNEC all_active_frontier(_graph);
    #endif

    #if defined(__USE_MULTICORE__)
    GraphAbstractionsMulticore graph_API(_graph);
    FrontierMulticore work_frontier(_graph);
    FrontierMulticore all_active_frontier(_graph);
    #endif

    VerticesArray<_T> prev_distances(_graph);

    graph_API.change_traversal_direction(SCATTER, _distances, work_frontier, all_active_frontier, prev_distances);

    Timer tm;
    tm.start();

    _T inf_val = std::numeric_limits<_T>::max() - MAX_WEIGHT;
    auto init_distances = [&_distances, _source_vertex, inf_val] __VGL_COMPUTE_ARGS__
    {
        if(src_id == _source_vertex)
            _distances[src_id] = 0;
        else
            _distances[src_id] = inf_val;
    };
    all_active_frontier.set_all_active();
    graph_API.compute(_graph, all_active_frontier, init_distances);

    work_frontier.clear();
    work_frontier.add_vertex(_source_vertex);

    int iterations_count = 0;
    while(work_frontier.size() > 0)
    {
        auto copy_distances = [&_distances, &prev_distances] __VGL_COMPUTE_ARGS__
        {
            prev_distances[src_id] = _distances[src_id];
        };
        graph_API.compute(_graph, all_active_frontier, copy_distances);

        #pragma omp parallel
        {
            auto edge_op_push = [&_distances, &_weights](int src_id, int dst_id, int local_edge_pos,
                            long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
            {
                _T weight = _weights[global_edge_pos];
                _T src_weight = _distances[src_id];

                if(_distances[dst_id] > src_weight + weight)
                {
                    _distances[dst_id] = src_weight + weight;
                }
            };

            graph_API.scatter(_graph, work_frontier, edge_op_push, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP,
                              edge_op_push, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);
        }

        auto changes_occurred = [&_distances, &prev_distances] (int src_id, int connections_count)->int
        {
            int result = NOT_IN_FRONTIER_FLAG;
            if(_distances[src_id] != prev_distances[src_id])
                result = IN_FRONTIER_FLAG;
            return result;
        };

        graph_API.generate_new_frontier(_graph, work_frontier, changes_occurred);

        iterations_count++;
    }

    tm.end();

    performance_stats.save_algorithm_performance_stats(tm.get_time(), _graph.get_edges_count(), iterations_count);
    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    performance_stats.print_algorithm_performance_stats("SSSP (Dijkstra, partial active)");
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
template <typename _T>
void SSSP::nec_dijkstra_all_active_push(VectCSRGraph &_graph,
                                        EdgesArray_Vect<_T> &_weights,
                                        VerticesArray<_T> &_distances,
                                        int _source_vertex)
{
    #if defined(__USE_NEC_SX_AURORA__)
    GraphAbstractionsNEC graph_API(_graph);
    FrontierNEC frontier(_graph);
    #endif

    #if defined(__USE_MULTICORE__)
    GraphAbstractionsMulticore graph_API(_graph);
    FrontierMulticore frontier(_graph);
    #endif

    graph_API.change_traversal_direction(SCATTER, _distances, frontier);

    Timer tm;
    tm.start();

    _T inf_val = std::numeric_limits<_T>::max() - MAX_WEIGHT;
    auto init_distances = [&_distances, _source_vertex, inf_val] __VGL_COMPUTE_ARGS__
    {
        if(src_id == _source_vertex)
            _distances[src_id] = 0;
        else
            _distances[src_id] = inf_val;
    };
    frontier.set_all_active();
    graph_API.compute(_graph, frontier, init_distances);

    int changes = 0, iterations_count = 0;
    do
    {
        changes = 0;
        iterations_count++;

        #pragma omp parallel shared(changes)
        {
            NEC_REGISTER_INT(was_changes, 0);

            _T *distances_ptr = _distances.get_ptr();
            auto edge_op_push = [&_distances, &_weights, &reg_was_changes, distances_ptr](int src_id, int dst_id, int local_edge_pos,
                            long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
            {
                _T weight = _weights[global_edge_pos];
                _T src_weight = distances_ptr[src_id];

                if(distances_ptr[dst_id] > src_weight + weight)
                {
                    distances_ptr[dst_id] = src_weight + weight;
                    reg_was_changes[vector_index] = 1;
                }
            };

            graph_API.scatter(_graph, frontier, edge_op_push, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP,
                              edge_op_push, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);

            #pragma omp critical
            {
                changes += register_sum_reduce(reg_was_changes);
            }
        }
    }
    while(changes);
    tm.end();

    performance_stats.save_algorithm_performance_stats(tm.get_time(), _graph.get_edges_count(), iterations_count);
    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    performance_stats.print_algorithm_performance_stats("SSSP (Dijkstra, all-active, push)");
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
template <typename _T>
void SSSP::nec_dijkstra_all_active_pull(VectCSRGraph &_graph,
                                        EdgesArray_Vect<_T> &_weights,
                                        VerticesArray<_T> &_distances,
                                        int _source_vertex)
{
    #if defined(__USE_NEC_SX_AURORA__)
    GraphAbstractionsNEC graph_API(_graph);
    FrontierNEC frontier(_graph);
    #endif

    #if defined(__USE_MULTICORE__)
    GraphAbstractionsMulticore graph_API(_graph);
    FrontierMulticore frontier(_graph);
    #endif

    VerticesArray<_T> prev_distances(_graph, GATHER);

    graph_API.change_traversal_direction(GATHER, _distances, frontier);

    Timer tm;
    tm.start();

    _T inf_val = std::numeric_limits<_T>::max() - MAX_WEIGHT;
    auto init_distances = [&_distances, _source_vertex, inf_val] __VGL_COMPUTE_ARGS__
    {
        if(src_id == _source_vertex)
            _distances[src_id] = 0;
        else
            _distances[src_id] = inf_val;
    };
    frontier.set_all_active();
    graph_API.compute(_graph, frontier, init_distances);

    int changes = 0, iterations_count = 0;
    do
    {
        changes = 0;
        iterations_count++;

        auto save_old_distances = [&_distances, &prev_distances] __VGL_COMPUTE_ARGS__
        {
            prev_distances[src_id] = _distances[src_id];
        };
        graph_API.compute(_graph, frontier, save_old_distances);

        #pragma omp parallel shared(changes)
        {
            NEC_REGISTER_INT(was_changes, 0);
            NEC_REGISTER_FLT(distances, 0);

            auto edge_op_pull = [&_distances, &_weights, &reg_distances](int src_id, int dst_id, int local_edge_pos,
                    long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
            {
                _T weight = _weights[global_edge_pos];
                _T dst_weight = _distances[dst_id];
                if(_distances[src_id] > dst_weight + weight)
                {
                    reg_distances[vector_index] = dst_weight + weight;
                }
            };

            auto preprocess = [&reg_distances, inf_val] (int src_id, int connections_count, int vector_index, DelayedWriteNEC &delayed_write)
            {
                #pragma _NEC ivdep
                for(int i = 0; i < VECTOR_LENGTH; i++)
                    reg_distances[i] = inf_val;
            };

            auto postprocess = [&_distances, &reg_distances, inf_val] (int src_id, int connections_count, int vector_index, DelayedWriteNEC &delayed_write)
            {
                _T min = inf_val;
                #pragma _NEC ivdep
                for(int i = 0; i < VECTOR_LENGTH; i++)
                    if(min > reg_distances[i])
                        min = reg_distances[i];
                if(_distances[src_id] > min)
                    _distances[src_id] = min;
            };

            auto edge_op_collective_pull = [&_distances, &_weights, &reg_distances]
                   (int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos,
                    int vector_index, DelayedWriteNEC &delayed_write)
            {
                _T weight = _weights[global_edge_pos];
                _T dst_weight = _distances[dst_id];
                if(reg_distances[vector_index] > dst_weight + weight)
                {
                    reg_distances[vector_index] = dst_weight + weight;
                }
            };

            auto preprocess_collective = [&reg_distances, inf_val] (int src_id, int connections_count, int vector_index, DelayedWriteNEC &delayed_write)
            {
                reg_distances[vector_index] = inf_val;
            };

            auto postprocess_collective = [&_distances, &reg_distances, inf_val] (int src_id, int connections_count, int vector_index, DelayedWriteNEC &delayed_write)
            {
                if(_distances[src_id] > reg_distances[vector_index])
                    _distances[src_id] = reg_distances[vector_index];
            };

            graph_API.enable_safe_stores();
            graph_API.gather(_graph, frontier, edge_op_pull, preprocess, postprocess,
                             edge_op_collective_pull, preprocess_collective, postprocess_collective);
            graph_API.disable_safe_stores();
        }

        auto reduce_changes = [&_distances, &prev_distances]__VGL_COMPUTE_ARGS__->int
        {
            int result = 0.0;
            if(prev_distances[src_id] != _distances[src_id])
            {
                result = 1;
            }
            return result;
        };
        changes = graph_API.reduce<int>(_graph, frontier, reduce_changes, REDUCE_SUM);
    }
    while(changes);
    tm.end();

    performance_stats.save_algorithm_performance_stats(tm.get_time(), _graph.get_edges_count(), iterations_count);
    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    performance_stats.print_algorithm_performance_stats("SSSP (Dijkstra, all-active, pull)");
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
template <typename _T>
void SSSP::nec_dijkstra(VectCSRGraph &_graph,
                        EdgesArray_Vect<_T> &_weights,
                        VerticesArray<_T> &_distances,
                        int _source_vertex,
                        AlgorithmFrontierType _frontier_type,
                        AlgorithmTraversalType _traversal_direction)
{
    if(_frontier_type == ALL_ACTIVE)
    {
        if(_traversal_direction == PUSH_TRAVERSAL)
            nec_dijkstra_all_active_push(_graph, _weights, _distances, _graph.reorder(_source_vertex, ORIGINAL, SCATTER));
        else if(_traversal_direction == PULL_TRAVERSAL)
            nec_dijkstra_all_active_pull(_graph, _weights, _distances, _graph.reorder(_source_vertex, ORIGINAL, GATHER));
    }
    else if(_frontier_type == PARTIAL_ACTIVE)
    {
        nec_dijkstra_partial_active(_graph, _weights, _distances, _graph.reorder(_source_vertex, ORIGINAL, SCATTER));
    }
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
template <typename _T>
void SSSP::nec_dijkstra(EdgesListGraph &_graph, EdgesArray_EL<_T> &_weights, VerticesArray<_T> &_distances,
                        int _source_vertex)
{
    #if defined(__USE_NEC_SX_AURORA__)
    GraphAbstractionsNEC graph_API(_graph);
    #endif

    #if defined(__USE_MULTICORE__)
    GraphAbstractionsMulticore graph_API(_graph);
    #endif

    Timer tm;
    tm.start();

    _T inf_val = std::numeric_limits<_T>::max() - MAX_WEIGHT;
    #pragma omp parallel for
    for(int i = 0; i < _graph.get_vertices_count(); i++)
    {
        _distances[i] = inf_val;
    }
    _distances[_source_vertex] = 0;

    int changes = 0, iterations_count = 0;
    do
    {
        changes = 0;
        iterations_count++;

        NEC_REGISTER_INT(was_changes, 0);

        auto edge_op = [&_distances, &_weights, &reg_was_changes](int src_id, int dst_id, int local_edge_pos,
                long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
        {
            _T weight = _weights[global_edge_pos];
            _T src_weight = _distances[src_id];

            if(_distances[dst_id] > src_weight + weight)
            {
                _distances[dst_id] = src_weight + weight;
                reg_was_changes[vector_index] = 1;
            }
        };

        graph_API.scatter(_graph, edge_op);

        changes += register_sum_reduce(reg_was_changes);
    }
    while(changes);
    tm.end();

    performance_stats.save_algorithm_performance_stats(tm.get_time(), _graph.get_edges_count(), iterations_count);
    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    performance_stats.print_algorithm_performance_stats("SSSP (Dijkstra, Edges List)");
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
template <typename _T>
void SSSP::nec_dijkstra(ShardedCSRGraph &_graph,
                        EdgesArray_Sharded<_T> &_weights,
                        VerticesArray<_T> &_distances,
                        int _source_vertex)
{
    #if defined(__USE_NEC_SX_AURORA__)
    GraphAbstractionsNEC graph_API(_graph);
    FrontierNEC frontier(_graph);
    #endif

    #if defined(__USE_MULTICORE__)
    GraphAbstractionsMulticore graph_API(_graph);
    FrontierMulticore frontier(_graph);
    #endif

    graph_API.attach_data(_distances);
    graph_API.change_traversal_direction(SCATTER); // TODO -- is it needed?

    Timer tm;
    tm.start();

    _T inf_val = std::numeric_limits<_T>::max() - MAX_WEIGHT;
    auto init_distances = [&_distances, _source_vertex, inf_val] __VGL_COMPUTE_ARGS__
    {
        if(src_id == _source_vertex)
            _distances[src_id] = 0;
        else
            _distances[src_id] = inf_val;
    };
    frontier.set_all_active();
    graph_API.compute(_graph, frontier, init_distances);

    int changes = 0, iterations_count = 0;
    do
    {
        changes = 0;
        iterations_count++;

        NEC_REGISTER_INT(was_changes, 0);

        auto edge_op_push = [&_distances, &_weights, &reg_was_changes, &changes](int src_id, int dst_id, int local_edge_pos,
                long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
        {
            _T weight = _weights[global_edge_pos];
            _T src_weight = _distances[src_id];

            if(_distances[dst_id] > src_weight + weight)
            {
                _distances[dst_id] = src_weight + weight;
                reg_was_changes[vector_index] = 1;
            }
        };

        graph_API.scatter(_graph, frontier, edge_op_push, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP,
                         edge_op_push, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);

        changes += register_sum_reduce(reg_was_changes);
    }
    while(changes);
    tm.end();

    performance_stats.save_algorithm_performance_stats(tm.get_time(), _graph.get_edges_count(), iterations_count);
    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    performance_stats.print_algorithm_performance_stats("SSSP (Dijkstra, sharded graph)");
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
