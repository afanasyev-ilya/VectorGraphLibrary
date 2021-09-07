#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
double BFS::fast_vgl_top_down(VGL_Graph &_graph,
                              VerticesArray<_T> &_levels,
                              int _source_vertex,
                              VGL_GRAPH_ABSTRACTIONS &_graph_API,
                              VGL_FRONTIER &_frontier)
{
    auto init_levels = [_levels, _source_vertex] __VGL_COMPUTE_ARGS__
    {
        if(src_id == _source_vertex)
            _levels[_source_vertex] = FIRST_LEVEL_VERTEX;
        else
            _levels[src_id] = UNVISITED_VERTEX;
    };
    _frontier.set_all_active();
    _graph_API.compute(_graph, _frontier, init_levels);

    _frontier.clear();
    _frontier.add_vertex(_source_vertex);

    int current_level = FIRST_LEVEL_VERTEX;
    while(_frontier.size() > 0)
    {
        auto edge_op = [_levels, current_level] __VGL_SCATTER_ARGS__
        {
            int src_level = _levels[src_id];
            int dst_level = _levels[dst_id];
            if((src_level == current_level) && (dst_level == UNVISITED_VERTEX))
            {
                _levels[dst_id] = current_level + 1;
            }
        };

        _graph_API.scatter(_graph, _frontier, edge_op);

        auto on_next_level = [_levels, current_level] __VGL_GNF_ARGS__ {
            int result = NOT_IN_FRONTIER_FLAG;
            if(_levels[src_id] == (current_level + 1))
                result = IN_FRONTIER_FLAG;
            return result;
        };

        _graph_API.generate_new_frontier(_graph, _frontier, on_next_level);

        current_level++;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
template <typename _T>
double BFS::vgl_top_down(VGL_Graph &_graph,
                         VerticesArray<_T> &_levels,
                         int _source_vertex)
{
    VGL_GRAPH_ABSTRACTIONS graph_API(_graph);
    VGL_FRONTIER frontier(_graph);

    graph_API.change_traversal_direction(SCATTER, _levels, frontier);

    #pragma omp parallel
    {};

    Timer tm;
    tm.start();
    fast_vgl_top_down(_graph, _levels, _source_vertex, graph_API, frontier);
    tm.end();

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    performance_stats.print_algorithm_performance_stats("BFS (Top-down, NEC/multicore)", tm.get_time(), _graph.get_edges_count());
    #endif

    return performance_stats.get_algorithm_performance(tm.get_time(), _graph.get_edges_count());
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


