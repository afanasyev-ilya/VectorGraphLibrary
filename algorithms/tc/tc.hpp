#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double TransitiveClosure::vgl_purdom(VGL_Graph &_graph,
                                     vector<pair<int,int>> &_vertex_pairs,
                                     vector<bool> &_answer)
{
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double TransitiveClosure::vgl_bfs_based(VGL_Graph &_graph,
                                        vector<pair<int,int>> &_vertex_pairs,
                                        vector<bool> &_answer)
{
    VerticesArray<int> levels(_graph);
    VGL_GRAPH_ABSTRACTIONS graph_API(_graph);
    VGL_FRONTIER frontier(_graph);

    graph_API.change_traversal_direction(SCATTER, levels, frontier);

    // run BFS _vertex_pairs.size() times
    Timer tm;
    tm.start();
    for(int i = 0; i < _vertex_pairs.size(); i++)
    {
        int source_vertex = _vertex_pairs[i].first;
        int end_vertex = _vertex_pairs[i].second;
        BFS::fast_vgl_top_down(_graph, levels, source_vertex, graph_API, frontier);
        if(levels[i] != UNVISITED_VERTEX)
        {
            _answer[i] = true;
        }
        else
        {
            _answer[i] = false;
        }
    }
    tm.end();

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    performance_stats.print_algorithm_performance_stats("TC", tm.get_time()/_vertex_pairs.size(), _graph.get_edges_count());
    #endif

    return performance_stats.get_algorithm_performance(tm.get_time()/_vertex_pairs.size(), _graph.get_edges_count());
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
