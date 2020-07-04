#pragma once

#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
bool nec_bfs(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
            int _source,
            int _sink,
            int *_parents,
            int *_levels,
            GraphPrimitivesNEC &_graph_API,
            FrontierNEC &_frontier)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);
    _frontier.set_all_active();

    auto init = [_parents, _source, _levels] (int src_id, int connections_count, int vector_index)
    {
        _parents[src_id] = -1;
        if(src_id == _source)
            _levels[_source] = FIRST_LEVEL_VERTEX;
        else
            _levels[src_id] = UNVISITED_VERTEX;
    };
    _graph_API.compute(_graph, _frontier, init);

    _frontier.add_vertex(_graph, _source);

    int current_level = FIRST_LEVEL_VERTEX;
    while(_frontier.size() > 0)
    {
        auto edge_op = [_levels, _parents, current_level, outgoing_weights](int src_id, int dst_id, int local_edge_pos,
                                 long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
        {
            int dst_level = _levels[dst_id];
            _TEdgeWeight weight = outgoing_weights[global_edge_pos];
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

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _TVertexValue, typename _TEdgeWeight>
_TEdgeWeight MF::nec_ford_fulkerson(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                    int _source, int _sink)
{
    GraphPrimitivesNEC graph_API;
    FrontierNEC frontier(_graph);

    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    int *parents;
    int *levels;
    MemoryAPI::allocate_array(&parents, vertices_count);
    MemoryAPI::allocate_array(&levels, vertices_count);

    #pragma omp parallel for
    for(int i = 0; i < edges_count; i++)
    {
        outgoing_weights[i] = 10;
    }

    int max_flow = 0;
    double bfs_time = 0;
    double reminder_time = 0, reminder_time1 = 0, reminder_time2 = 0;

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
        int path_flow = INT_MAX;
        int path_length = 0;
        for (int v = _sink; v != _source; v = parents[v])
        {
            int u = parents[v];
            int current_weight = get_weight(_graph, u, v);
            path_flow = min(path_flow, current_weight);
        }
        t2 = omp_get_wtime();
        reminder_time1 += t2 - t1;

        t1 = omp_get_wtime();
        for (int v = _sink; v != _source; v = parents[v])
        {
            int u = parents[v];
            minus_weight(_graph, u, v, path_flow);
            plus_weight(_graph, v, u, path_flow);
        }
        max_flow += path_flow;
        t2 = omp_get_wtime();
        reminder_time2 += t2 - t1;

        iterations_count++;
    }

    reminder_time = reminder_time1 + reminder_time2;

    cout << "iterations done: " << iterations_count << endl;
    cout << "bfs time: " << bfs_time*1000.0 << " ms" << endl;
    cout << "reminder time: " << reminder_time*1000.0 << " ms" << endl;
    cout << "reminder1 time: " << reminder_time1*1000.0 << " ms" << endl;
    cout << "reminder2 time: " << reminder_time2*1000.0 << " ms" << endl;
    cout << "average bfs perf: " << edges_count / ((bfs_time/iterations_count)*1e6) << " MTEPS" << endl;
    cout << "wall perf: " << edges_count / ((bfs_time + reminder_time)*1e6) << " MTEPS" << endl;

    MemoryAPI::free_array(parents);
    MemoryAPI::free_array(levels);

    // Return the overall flow
    return max_flow;
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
