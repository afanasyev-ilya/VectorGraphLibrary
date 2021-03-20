#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__

bool MF::nec_bfs(UndirectedCSRGraph &_graph,
                 int _source,
                 int _sink,
                 int *_parents,
                 int *_levels,
                 GraphPrimitivesNEC &_graph_API,
                 FrontierNEC &_frontier)
{
    LOAD_UNDIRECTED_CSR_GRAPH_DATA(_graph);
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
    _frontier.add_vertex(_graph, _source);

    int current_level = FIRST_LEVEL_VERTEX;
    while(_frontier.size() > 0)
    {
        auto edge_op = [_levels, _parents, current_level, adjacent_weights](int src_id, int dst_id, int local_edge_pos,
                                 long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
        {
            int dst_level = _levels[dst_id];
            _TEdgeWeight weight = adjacent_weights[global_edge_pos];
            if((dst_level == UNVISITED_VERTEX) && (weight > 0))
            {
                _levels[dst_id] = current_level + 1;
                _parents[dst_id] = src_id;
            }
        };

        auto on_next_level = [_levels, current_level] (int src_id)->int
        {
            int result = NOT_IN_FRONTIER_FLAG;
            if(_levels[src_id] == (current_level + 1))
                result = IN_FRONTIER_FLAG;
            return result;
        };

        _graph_API.advance(_graph, _frontier, _frontier, edge_op, on_next_level);

        current_level++;
    }

    if(_levels[_sink] == UNVISITED_VERTEX)
        return false;
    else
        return true;
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void construct_path(int _source, int _sink, int *_parents, int *_path, int &_path_length)
{
    int path_pos = 0;
    for (int v = _sink; v != _source; v = _parents[v])
    {
        int u = _parents[v];
        _path[path_pos] = v;
        path_pos++;
    }
    _path_length = path_pos;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__

_TEdgeWeight MF::nec_ford_fulkerson(UndirectedCSRGraph &_graph,
                                    int _source, int _sink)
{
    GraphPrimitivesNEC graph_API;
    FrontierNEC frontier(_graph);

    LOAD_UNDIRECTED_CSR_GRAPH_DATA(_graph);

    int *parents;
    int *levels;
    int *path;
    MemoryAPI::allocate_array(&parents, vertices_count);
    MemoryAPI::allocate_array(&levels, vertices_count);
    MemoryAPI::allocate_array(&path, vertices_count);
    int *flows = adjacent_weights;

    #pragma omp parallel for
    for(int i = 0; i < edges_count; i++)
    {
        flows[i] = 10;
    }

    int max_flow = 0;
    double bfs_time = 0, reminder_time = 0;
    double avg_path_length = 0;

    int iterations_count = 0;
    while(true)
    {
        double t1 = omp_get_wtime();
        bool sink_reached = nec_bfs(_graph, _source, _sink, parents, levels, graph_API, frontier);
        double t2 = omp_get_wtime();
        bfs_time += t2 - t1;

        if(!sink_reached)
            break;

        t1 = omp_get_wtime();
        // construct path
        int path_length = 0;
        construct_path(_source, _sink, parents, path, path_length);
        avg_path_length += path_length;

        // calculate current flow
        int path_flow = std::numeric_limits<std::int32_t>::max();
        for (int i = 0; i < path_length; i++)
        {
            int v = path[i];
            int u = parents[v];
            int current_flow = _graph.get_edge_data(flows, u, v);
            path_flow = min(path_flow, current_flow);
        }

        // update weights
        for (int i = 0; i < path_length; i++)
        {
            int v = path[i];
            int u = parents[v];
            _graph.get_edge_data(flows, u, v) -= path_flow;
            //_graph.get_edge_data(flows, v, u) += path_flow;
        }
        max_flow += path_flow;
        iterations_count++;

        t2 = omp_get_wtime();
        reminder_time += t2 - t1;
    }

    avg_path_length /= iterations_count;

    cout << "iterations done: " << iterations_count << endl;
    cout << "bfs time: " << bfs_time*1000.0 << " ms" << endl;
    cout << "reminder time: " << reminder_time*1000.0 << " ms" << endl;
    cout << "average bfs perf: " << edges_count / ((bfs_time/iterations_count)*1e6) << " MTEPS" << endl;
    cout << "average path length: " << avg_path_length << endl;
    cout << "wall perf: " << edges_count / ((bfs_time + reminder_time)*1e6) << " MTEPS" << endl;

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    performance_stats.print_algorithm_performance_stats("MF nec_ford_fulkerson", bfs_time + reminder_time, edges_count, iterations_count);
    #endif

    MemoryAPI::free_array(parents);
    MemoryAPI::free_array(levels);
    MemoryAPI::free_array(path);

    // Return the overall flow
    return max_flow;
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
