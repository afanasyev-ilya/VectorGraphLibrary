#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
template <typename _T>
void BFS::nec_top_down(VectCSRGraph &_graph,
                       VerticesArray<_T> &_levels,
                       int _source_vertex)
{
    VGL_GRAPH_ABSTRACTIONS graph_API(_graph);
    VGL_FRONTIER frontier(_graph);

    graph_API.change_traversal_direction(SCATTER, _levels, frontier);

    _source_vertex = _graph.reorder(_source_vertex, ORIGINAL, SCATTER);

    #pragma omp parallel
    {};

    auto init_levels = [&_levels, _source_vertex] __VGL_COMPUTE_ARGS__
    {
        if(src_id == _source_vertex)
            _levels[_source_vertex] = FIRST_LEVEL_VERTEX;
        else
            _levels[src_id] = UNVISITED_VERTEX;
    };
    frontier.set_all_active();
    graph_API.compute(_graph, frontier, init_levels);

    frontier.clear();
    frontier.add_vertex(_source_vertex);

    Timer tm;
    tm.start();

    int current_level = FIRST_LEVEL_VERTEX;
    while(frontier.size() > 0)
    {
        auto edge_op = [&_levels, &current_level](int src_id, int dst_id, int local_edge_pos,
                long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
        {
            int src_level = _levels[src_id];
            int dst_level = _levels[dst_id];
            if((src_level == current_level) && (dst_level == UNVISITED_VERTEX))
            {
                _levels[dst_id] = current_level + 1;
            }
        };

        graph_API.scatter(_graph, frontier, edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP,
                          edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);

        auto on_next_level = [&_levels, current_level] (int src_id, int connections_count)->int
        {
            int result = NOT_IN_FRONTIER_FLAG;
            if(_levels[src_id] == (current_level + 1))
                result = IN_FRONTIER_FLAG;
            return result;
        };

        graph_API.generate_new_frontier(_graph, frontier, on_next_level);

        current_level++;
    }
    tm.end();

    performance_stats.save_algorithm_performance_stats(tm.get_time(), _graph.get_edges_count());
    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    performance_stats.print_algorithm_performance_stats("BFS (Top-down, NEC/multicore)");
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


