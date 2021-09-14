#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
bool MF::mf_bfs(VGL_Graph &_graph,
                EdgesArray<_T> &_weights,
                int _source,
                int _sink,
                VerticesArray<int> &_parents,
                VerticesArray<int> &_levels,
                VGL_GRAPH_ABSTRACTIONS &_graph_API,
                VGL_FRONTIER &_frontier)
{
    _frontier.set_all_active();

    auto init = [_parents, _source, _levels] __VGL_COMPUTE_ARGS__
    {
        _parents[src_id] = -1;
        if(src_id == _source)
            _levels[_source] = FIRST_LEVEL_VERTEX;
        else
            _levels[src_id] = UNVISITED_VERTEX;
    };
    _graph_API.compute(_graph, _frontier, init);

    _frontier.clear();
    _frontier.add_vertex(_source);

    int current_level = FIRST_LEVEL_VERTEX;
    while(_frontier.size() > 0)
    {
        auto edge_op = [_levels, _parents, current_level, _weights] __VGL_ADVANCE_ARGS__
        {
            int dst_level = _levels[dst_id];
            _T weight = _weights[global_edge_pos];
            if((dst_level == UNVISITED_VERTEX) && (weight > 0))
            {
                _levels[dst_id] = current_level + 1;
                _parents[dst_id] = src_id;
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

    if(_levels[_sink] == UNVISITED_VERTEX)
        return false;
    else
        return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
double MF::vgl_ford_fulkerson(VGL_Graph &_graph, EdgesArray<_T> &_flows, int _source, int _sink, _T &_max_flow)
{
    VGL_GRAPH_ABSTRACTIONS graph_API(_graph);
    VGL_FRONTIER frontier(_graph);

    int vertices_count = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count();

    VerticesArray<int> parents(_graph);
    VerticesArray<int> levels(_graph);

    graph_API.change_traversal_direction(SCATTER, frontier);

    Timer tm;
    tm.start();

    int avg_path_length = 0;
    int iterations_count = 0;
    while(mf_bfs(_graph, _flows, _source, _sink, parents, levels, graph_API, frontier))
    {
        // Find minimum residual capacity of the edges along the
        // path filled by BFS. Or we can say find the maximum flow
        // through the path found.
        _T path_flow = std::numeric_limits<_T>::max();
        int path_length = 0;
        for (int v = _sink; v != _source; v = parents[v])
        {
            int u = parents[v];
            _T current_weight = get_flow(_graph, _flows, u, v);
            path_flow = min(path_flow, current_weight);
            path_length++;
        }
        avg_path_length += path_length;

        /// update residual capacities of the edges and reverse edges
        // along the path
        for (int v = _sink; v != _source; v = parents[v])
        {
            int u = parents[v];
            subtract_flow(_graph, _flows, u, v, path_flow);
            add_flow(_graph, _flows, v, u, path_flow);
        }

        // Add path flow to overall flow
        _max_flow += path_flow;
        iterations_count++;
    }
    avg_path_length /= iterations_count;

    tm.end();

    cout << "iterations done: " << iterations_count << endl;
    cout << "average path length: " << avg_path_length << endl;

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    performance_stats.print_algorithm_performance_stats("MF (Ford-Fulkerson)", tm.get_time(), _graph.get_edges_count());
    #endif

    return performance_stats.get_algorithm_performance(tm.get_time(), _graph.get_edges_count());
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
