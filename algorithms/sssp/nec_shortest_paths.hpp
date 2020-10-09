#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _T>
void SSSP::nec_dijkstra_partial_active(VectCSRGraph &_graph,
                                       EdgesArrayNec<_T> &_weights,
                                       VerticesArrayNec<_T> &_distances,
                                       int _source_vertex)
{
    GraphAbstractionsNEC graph_API(_graph, SCATTER);
    FrontierNEC work_frontier(_graph, SCATTER);
    FrontierNEC all_active_frontier(_graph, SCATTER);

    graph_API.change_traversal_direction(SCATTER);
    all_active_frontier.set_all_active();

    VerticesArrayNec<_T> prev_distances(_graph, SCATTER);

    _source_vertex = _graph.reorder(_source_vertex, ORIGINAL, SCATTER);

    if(!graph_API.have_correct_direction(_distances, prev_distances))
    {
        throw "Error: incorrect direction of vertex array in SSSP::nec_dijkstra_partial_active";
    }

    Timer tm;
    tm.start();

    auto init_distances = [&_distances, _source_vertex] (int src_id, int connections_count, int vector_index)
    {
        if(src_id == _source_vertex)
            _distances[_source_vertex] = 0;
        else
            _distances[src_id] = FLT_MAX;
    };

    graph_API.compute(_graph, all_active_frontier, init_distances); // init distances with all-active frontier

    auto is_source = [_source_vertex] (int src_id)->int
    {
        int result = NOT_IN_FRONTIER_FLAG;
        if(src_id == _source_vertex)
            result = IN_FRONTIER_FLAG;
        return result;
    };

    graph_API.generate_new_frontier(_graph, work_frontier, is_source);

    int iterations_count = 0;
    while(work_frontier.size() > 0)
    {
        auto copy_distances = [&_distances, &prev_distances] (int src_id, int connections_count, int vector_index)
        {
            prev_distances[src_id] = _distances[src_id];
        };
        graph_API.compute(_graph, all_active_frontier, copy_distances);

        #pragma omp parallel
        {
            auto edge_op_push = [&_distances, &_weights](int src_id, int dst_id, int local_edge_pos,
                            long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
            {
                _T weight = _weights.get(global_edge_pos);
                _T src_weight = _distances[src_id];

                if(_distances[dst_id] > src_weight + weight)
                {
                    _distances[dst_id] = src_weight + weight;
                }
            };

            graph_API.scatter(_graph, work_frontier, edge_op_push, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP,
                             edge_op_push, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);
        }

        auto changes_occurred = [&_distances, &prev_distances] (int src_id)->int
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
    PerformanceStats::print_performance_stats("partial active sssp (dijkstra)", tm.get_time(),
                                              _graph.get_edges_count(), iterations_count);
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*#ifdef __USE_NEC_SX_AURORA__
void SSSP::nec_bellamn_ford(EdgesListGraph &_graph,
                            _TEdgeWeight *_distances,
                            int _source_vertex)
{
    double t1 = omp_get_wtime();
    LOAD_EDGES_LIST_GRAPH_DATA(_graph);

    auto init_distances = [_distances, _source_vertex] (int src_id, int connections_count, int vector_index)
    {
        if(src_id == _source_vertex)
            _distances[_source_vertex] = 0;
        else
            _distances[src_id] = FLT_MAX;
    };
    graph_API.compute(_graph, init_distances);

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
#endif*/

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _T>
void SSSP::nec_dijkstra_all_active_push(VectCSRGraph &_graph,
                                        EdgesArrayNec<_T> &_weights,
                                        VerticesArrayNec<_T> &_distances,
                                        int _source_vertex)
{
    GraphAbstractionsNEC graph_API(_graph, SCATTER);
    FrontierNEC frontier(_graph, SCATTER);

    graph_API.change_traversal_direction(SCATTER);
    frontier.set_all_active();

    _source_vertex = _graph.reorder(_source_vertex, ORIGINAL, SCATTER);

    Timer tm;
    tm.start();

    if(!graph_API.have_correct_direction(_distances))
    {
        throw "Error: incorrect direction of vertex array in SSSP::nec_dijkstra_all_active_push";
    }

    int vect_csr_source_vertex = _source_vertex;
    auto init_distances = [&_distances, vect_csr_source_vertex] (int src_id, int connections_count, int vector_index)
    {
        if(src_id == vect_csr_source_vertex)
            _distances[src_id] = 0;
        else
            _distances[src_id] = FLT_MAX;
    };
    graph_API.compute(_graph, frontier, init_distances);

    int changes = 0, iterations_count = 0;
    do
    {
        changes = 0;
        iterations_count++;

        #pragma omp parallel shared(changes)
        {
            NEC_REGISTER_INT(was_changes, 0);

            auto edge_op_push = [&_distances, &_weights, &reg_was_changes, &changes](int src_id, int dst_id, int local_edge_pos,
                            long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
            {
                _T weight = _weights.get(global_edge_pos);
                _T src_weight = _distances[src_id];

                if(_distances[dst_id] > src_weight + weight)
                {
                    _distances[dst_id] = src_weight + weight;
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

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    PerformanceStats::print_performance_stats("sssp (dijkstra, all-active, push)", tm.get_time(),
                                              _graph.get_edges_count(), iterations_count);
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _T>
void SSSP::nec_dijkstra_all_active_pull(VectCSRGraph &_graph,
                                        EdgesArrayNec<_T> &_weights,
                                        VerticesArrayNec<_T> &_distances,
                                        int _source_vertex)
{
    GraphAbstractionsNEC graph_API(_graph, GATHER);
    FrontierNEC frontier(_graph, GATHER);

    graph_API.change_traversal_direction(GATHER);
    frontier.set_all_active();

    _source_vertex = _graph.reorder(_source_vertex, ORIGINAL, GATHER);

    if(!graph_API.have_correct_direction(_distances))
    {
        throw "Error: incorrect direction of vertex array in SSSP::nec_dijkstra_all_active_pull";
    }

    Timer tm;
    tm.start();

    int vect_csr_source_vertex = _source_vertex;
    auto init_distances = [&_distances, vect_csr_source_vertex] (int src_id, int connections_count, int vector_index)
    {
        if(src_id == vect_csr_source_vertex)
            _distances[src_id] = 0;
        else
            _distances[src_id] = FLT_MAX;
    };
    graph_API.compute(_graph, frontier, init_distances);

    int changes = 0, iterations_count = 0;
    do
    {
        changes = 0;
        iterations_count++;

        #pragma omp parallel shared(changes)
        {
            NEC_REGISTER_INT(was_changes, 0);
            NEC_REGISTER_FLT(distances, 0);

            auto edge_op_pull = [&_distances, &_weights, &reg_was_changes, &reg_distances](int src_id, int dst_id, int local_edge_pos,
                    long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
            {
                _T weight = 1;//_T weight = _weights.get(global_edge_pos);
                _T dst_weight = _distances[dst_id];
                if(_distances[src_id] > dst_weight + weight)
                {
                    _distances[src_id] = dst_weight + weight;
                    reg_was_changes[vector_index] = 1;
                }
            };

            auto vertex_preprocess_op = [&_distances, &reg_distances]
                    (int src_id, int connections_count, int vector_index, DelayedWriteNEC &delayed_write)
            {

            };

            auto vertex_postprocess_op = [&_distances, &reg_distances]
                        (int src_id, int connections_count, int vector_index, DelayedWriteNEC &delayed_write)
            {

            };

            auto edge_op_collective_pull = [&_distances, &_weights, &reg_was_changes]
                   (int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos,
                    int vector_index, DelayedWriteNEC &delayed_write)
            {
                _T weight = _weights.get(global_edge_pos);
                _T dst_weight = _distances[dst_id];
                if(_distances[src_id] > dst_weight + weight)
                {
                    _distances[src_id] = dst_weight + weight;
                    reg_was_changes[vector_index] = 1;
                }
            };

            graph_API.gather(_graph, frontier, edge_op_pull, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP,
                             edge_op_collective_pull, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);

            #pragma omp critical
            {
                changes += register_sum_reduce(reg_was_changes);
            }
        }
    }
    while(changes);

    tm.end();

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    PerformanceStats::print_performance_stats("sssp (dijkstra, all-active, pull)", tm.get_time(),
                                              _graph.get_edges_count(), iterations_count);
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _T>
void SSSP::nec_dijkstra(VectCSRGraph &_graph,
                        EdgesArrayNec<_T> &_weights,
                        VerticesArrayNec<_T> &_distances,
                        int _source_vertex,
                        AlgorithmFrontierType _frontier_type,
                        AlgorithmTraversalType _traversal_direction)
{
    if(_frontier_type == ALL_ACTIVE)
    {
        if(_traversal_direction == PUSH_TRAVERSAL)
            nec_dijkstra_all_active_push(_graph, _weights, _distances, _source_vertex);
        else if(_traversal_direction == PULL_TRAVERSAL)
            nec_dijkstra_all_active_pull(_graph, _weights, _distances, _source_vertex);
    }
    else if(_frontier_type == PARTIAL_ACTIVE)
    {
        nec_dijkstra_partial_active(_graph, _weights, _distances, _source_vertex);
    }
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
