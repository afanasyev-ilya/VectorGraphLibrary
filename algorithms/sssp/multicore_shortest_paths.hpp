#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void SSSP::multicore_dijkstra(VectCSRGraph &_graph,
                              EdgesArray<_T> &_weights,
                              VerticesArray<_T> &_distances,
                              int _source_vertex)
{
    GraphAbstractionsNEC graph_API(_graph);
    FrontierNEC frontier(_graph);
    graph_API.change_traversal_direction(SCATTER, _distances, frontier);

    Timer tm;
    tm.start();

    _T inf_val = std::numeric_limits<_T>::max() - MAX_WEIGHT;
    auto init_distances = [&_distances, _source_vertex, inf_val] (int src_id, int connections_count, int vector_index)
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
    PerformanceStats::print_algorithm_performance_stats("SSSP (Dijkstra, all-active, push, Multicore)", tm.get_time(),
                                                        _graph.get_edges_count(), iterations_count);
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
