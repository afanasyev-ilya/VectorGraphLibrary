#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
template <typename _T>
double SSSP::vgl_dijkstra_partial_active(VGL_Graph &_graph,
                                         EdgesArray<_T> &_weights,
                                         VerticesArray<_T> &_distances,
                                         int _source_vertex)
{
    VGL_GRAPH_ABSTRACTIONS graph_API(_graph);
    VGL_FRONTIER work_frontier(_graph);
    VGL_FRONTIER all_active_frontier(_graph);

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

        auto edge_op_push = [&_distances, &_weights] __VGL_SCATTER_ARGS__
        {
            _T weight = _weights[global_edge_pos];
            _T src_weight = _distances[src_id];
            if(_distances[dst_id] > src_weight + weight)
            {
                _distances[dst_id] = src_weight + weight;
            }
        };

        graph_API.scatter(_graph, work_frontier, edge_op_push);

        auto changes_occurred = [&_distances, &prev_distances] __VGL_GNF_ARGS__
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

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    performance_stats.print_algorithm_performance_stats("SSSP (Dijkstra, partial active)", tm.get_time(), _graph.get_edges_count());
    #endif

    return performance_stats.get_algorithm_performance(tm.get_time(), _graph.get_edges_count());
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
template <typename _T>
double SSSP::vgl_dijkstra_all_active_push(VGL_Graph &_graph,
                                          EdgesArray<_T> &_weights,
                                          VerticesArray<_T> &_distances,
                                          int _source_vertex)
{
    VGL_GRAPH_ABSTRACTIONS graph_API(_graph, SCATTER);
    VGL_FRONTIER frontier(_graph, SCATTER);
    VerticesArray<_T> prev_distances(_graph, SCATTER);

    graph_API.change_traversal_direction(SCATTER, _distances, prev_distances, frontier);

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

        auto edge_op_push = [&_distances, &_weights] __VGL_SCATTER_ARGS__
        {
            _T weight = _weights[global_edge_pos];
            _T src_weight = _distances[src_id];
            if(_distances[dst_id] > src_weight + weight)
            {
                _distances[dst_id] = src_weight + weight;
            }
        };

        graph_API.scatter(_graph, frontier, edge_op_push);

        #ifdef __USE_MPI__
        auto min_op = [](_T _a, _T _b)->_T
        {
            return vect_min(_a, _b);
        };
        graph_API.exchange_vertices_array(EXCHANGE_RECENTLY_CHANGED, _graph, _distances, prev_distances, min_op);
        #endif

        auto reduce_changes = [&_distances, &prev_distances]__VGL_REDUCE_INT_ARGS__
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

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    performance_stats.print_algorithm_performance_stats("SSSP (Dijkstra, all-active, push)", tm.get_time(), _graph.get_edges_count());
    #endif

    return performance_stats.get_algorithm_performance(tm.get_time(), _graph.get_edges_count());
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
template <typename _T>
double SSSP::vgl_dijkstra_all_active_pull(VGL_Graph &_graph,
                                          EdgesArray<_T> &_weights,
                                          VerticesArray<_T> &_distances,
                                          int _source_vertex)
{
    VGL_GRAPH_ABSTRACTIONS graph_API(_graph, GATHER);
    VGL_FRONTIER frontier(_graph, GATHER);

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
            VEC_REGISTER_FLT(distances, 0);

            auto edge_op_pull = [&_distances, &_weights, &reg_distances] __VGL_GATHER_ARGS__
            {
                _T weight = _weights[global_edge_pos];
                _T dst_weight = _distances[dst_id];
                if(_distances[src_id] > dst_weight + weight)
                {
                    reg_distances[vector_index] = dst_weight + weight;
                }
            };

            auto preprocess = [&reg_distances, inf_val] (int src_id, int connections_count, int vector_index)
            {
                #pragma _NEC ivdep
                for(int i = 0; i < VECTOR_LENGTH; i++)
                    reg_distances[i] = inf_val;
            };

            auto postprocess = [&_distances, &reg_distances, inf_val] (int src_id, int connections_count, int vector_index)
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
                    int vector_index)
            {
                _T weight = _weights[global_edge_pos];
                _T dst_weight = _distances[dst_id];
                if(reg_distances[vector_index] > dst_weight + weight)
                {
                    reg_distances[vector_index] = dst_weight + weight;
                }
            };

            auto preprocess_collective = [&reg_distances, inf_val] (int src_id, int connections_count, int vector_index)
            {
                reg_distances[vector_index] = inf_val;
            };

            auto postprocess_collective = [&_distances, &reg_distances, inf_val] (int src_id, int connections_count, int vector_index)
            {
                if(_distances[src_id] > reg_distances[vector_index])
                    _distances[src_id] = reg_distances[vector_index];
            };

            graph_API.enable_safe_stores();
            graph_API.gather(_graph, frontier, edge_op_pull, preprocess, postprocess,
                             edge_op_collective_pull, preprocess_collective, postprocess_collective);
            graph_API.disable_safe_stores();
        }

        #ifdef __USE_MPI__
        graph_API.exchange_vertices_array(EXCHANGE_PRIVATE_DATA, _graph, _distances);
        #endif

        auto reduce_changes = [&_distances, &prev_distances] __VGL_REDUCE_INT_ARGS__
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

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    performance_stats.print_algorithm_performance_stats("SSSP (Dijkstra, all-active, pull)", tm.get_time(), _graph.get_edges_count());
    #endif

    return performance_stats.get_algorithm_performance(tm.get_time(), _graph.get_edges_count());
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
template <typename _T>
double SSSP::vgl_dijkstra(VGL_Graph &_graph,
                          EdgesArray<_T> &_weights,
                          VerticesArray<_T> &_distances,
                          int _source_vertex,
                          AlgorithmFrontierType _frontier_type,
                          AlgorithmTraversalType _traversal_direction)
{
    if(_frontier_type == ALL_ACTIVE)
    {
        if(_traversal_direction == PUSH_TRAVERSAL)
            return vgl_dijkstra_all_active_push(_graph, _weights, _distances, _source_vertex);
        else if(_traversal_direction == PULL_TRAVERSAL)
            return vgl_dijkstra_all_active_pull(_graph, _weights, _distances, _source_vertex);
    }
    else if(_frontier_type == PARTIAL_ACTIVE)
    {
        return vgl_dijkstra_partial_active(_graph, _weights, _distances, _source_vertex);
    }
    return 0;
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
