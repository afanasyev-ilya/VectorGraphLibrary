/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _T>
void BFS::vgl_top_down(VGL_Graph &_graph,
                       VerticesArray<_T> &_levels,
                       int _source_vertex)
{
    _graph.move_to_device();
    _levels.move_to_device();

    VGL_GRAPH_ABSTRACTIONS graph_API(_graph);
    VGL_FRONTIER frontier(_graph);
    graph_API.change_traversal_direction(SCATTER, _levels, frontier);

    _source_vertex = _graph.reorder(_source_vertex, ORIGINAL, SCATTER);

    auto init_levels = [_levels, _source_vertex] __VGL_COMPUTE_ARGS__
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

    /*int current_level = FIRST_LEVEL_VERTEX;
    while(frontier.size() > 0)
    {
        auto edge_op = [_levels, current_level] __device__ (int src_id, int dst_id, int local_edge_pos,
                long long int global_edge_pos, int vector_index)
        {
            _T dst_level = _levels[dst_id];
            if(dst_level == UNVISITED_VERTEX)
            {
                _levels[dst_id] = current_level + 1;
            }
        };

        graph_API.scatter(_graph, frontier, edge_op);

        auto on_next_level = [_levels, current_level] __device__ __VGL_GNF_ARGS__
        {
            int result = NOT_IN_FRONTIER_FLAG;
            if(_levels[src_id] == (current_level + 1))
                result = IN_FRONTIER_FLAG;
            return result;
        };

        graph_API.generate_new_frontier(_graph, frontier, on_next_level);

        current_level++;
    }*/
    tm.end();

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    performance_stats.print_algorithm_performance_stats("BFS (Top-down, GPU)", tm.get_time(), _graph.get_edges_count());
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
