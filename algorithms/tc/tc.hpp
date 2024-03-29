#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double TransitiveClosure::vgl_purdoms(VGL_Graph &_graph,
                                      vector<pair<int,int>> &_vertex_pairs,
                                      vector<int> &_answer)
{
    Timer tm;
    tm.start();

    // compute SCC for original graph
    VerticesArray<int> components(_graph);
    SCC::vgl_forward_backward(_graph, components);

    // compute answer for obvious pairs
    vector<pair<pair<int, int>, int>> reminder_pairs;
    for(int i = 0; i < _vertex_pairs.size(); i++)
    {
        int source_vertex = _vertex_pairs[i].first;
        int end_vertex = _vertex_pairs[i].second;

        if(components[source_vertex] == components[end_vertex])
        {
            _answer[i] = true;
        }
        else
        {
            reminder_pairs.push_back(make_pair(_vertex_pairs[i], i));
        }
    }

    // if all pairs processed -- early terminate
    if(reminder_pairs.size() == 0)
    {
        tm.end();

        #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
        performance_stats.print_algorithm_performance_stats("TC", tm.get_time()/_vertex_pairs.size(), _graph.get_edges_count());
        #endif
        return performance_stats.get_algorithm_performance(tm.get_time()/_vertex_pairs.size(), _graph.get_edges_count());
    }

    // compute intermediate representation graph
    EdgesArray<int> new_src_ids(_graph);
    EdgesArray<int> new_dst_ids(_graph);

    VGL_GRAPH_ABSTRACTIONS graph_API(_graph);
    VGL_FRONTIER frontier(_graph);
    graph_API.change_traversal_direction(SCATTER, frontier, components);

    // filter edges
    frontier.set_all_active();
    long long edges_count = _graph.get_edges_count();
    auto edge_op = [components, new_src_ids, new_dst_ids] __VGL_SCATTER_ARGS__ {
        int src_comp = components[src_id];
        int dst_comp = components[dst_id];
        if(src_comp != dst_comp)
        {
            new_src_ids[global_edge_pos] = src_comp;
            new_dst_ids[global_edge_pos] = dst_comp;
        }
        else
        {
            new_src_ids[global_edge_pos] = -1;
            new_dst_ids[global_edge_pos] = -1;
        }
    };
    graph_API.scatter(_graph, frontier, edge_op);

    auto merge_edges_array_data = [new_src_ids, new_dst_ids] (int val1, int val2) {
        if(val1 != 0)
            return val1;
        if(val2 != 0)
            return val2;
        return 0;
    };
    new_src_ids.finalize_advance(merge_edges_array_data);
    new_dst_ids.finalize_advance(merge_edges_array_data);

    int *edges_buffer;
    int *edge_indexes;
    MemoryAPI::allocate_array(&edges_buffer, edges_count);
    MemoryAPI::allocate_array(&edge_indexes, edges_count);

    auto edge_connects_different_components = [new_src_ids, new_dst_ids] __VGL_COPY_IF_INDEXES_ARGS__
    {
        if((new_src_ids[idx] != -1) && (new_dst_ids[idx] != -1) && (new_src_ids[idx]!= new_dst_ids[idx]))
        {
            return 1;
        }
        else
            return 0;
    };
    int new_edges_count = ParallelPrimitives::copy_if_indexes(edge_connects_different_components,
                                                              edge_indexes, edges_count, edges_buffer, edges_count, 0);

    MemoryAPI::free_array(edges_buffer);

    auto max_component_num = [components] __VGL_REDUCE_INT_ARGS__
    {
        return components[src_id];
    };
    int new_vertices_count = graph_API.reduce<int>(_graph, frontier, max_component_num, REDUCE_MAX);

    cout << "new vertices count: " << new_vertices_count << endl;
    cout << "new edges count: " << new_edges_count << endl;

    EdgesContainer tmp_edges;
    tmp_edges.resize(new_vertices_count, new_edges_count);

    openmp_reorder_gather_copy(new_src_ids.get_ptr(), tmp_edges.get_src_ids(), edge_indexes, new_edges_count);
    openmp_reorder_gather_copy(new_dst_ids.get_ptr(), tmp_edges.get_dst_ids(), edge_indexes, new_edges_count);

    MemoryAPI::free_array(edge_indexes);

    VGL_Graph ir_graph(EDGES_LIST_GRAPH);
    ir_graph.import(tmp_edges);

    VerticesArray<int> ir_levels(ir_graph);
    VGL_GRAPH_ABSTRACTIONS ir_graph_API(ir_graph);
    VGL_FRONTIER ir_frontier(ir_graph);

    ir_graph_API.change_traversal_direction(SCATTER, ir_levels, ir_frontier);

    // compute answer for reminder vertices
    for(int idx = 0; idx < reminder_pairs.size(); idx++)
    {
        pair<int, int> vertex_pair = reminder_pairs[idx].first;
        int result_index = reminder_pairs[idx].second;

        int source_vertex = vertex_pair.first;
        int end_vertex = vertex_pair.second;

        BFS::fast_vgl_top_down(ir_graph, ir_levels, components[source_vertex], ir_graph_API, ir_frontier);
        if(ir_levels[components[end_vertex]] != UNVISITED_VERTEX)
        {
            _answer[result_index] = true;
        }
        else
        {
            _answer[result_index] = false;
        }
    }

    tm.end();

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    performance_stats.print_algorithm_performance_stats("TC", tm.get_time()/_vertex_pairs.size(), _graph.get_edges_count());
    #endif

    return performance_stats.get_algorithm_performance(tm.get_time()/_vertex_pairs.size(), _graph.get_edges_count());
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double TransitiveClosure::vgl_bfs_based(VGL_Graph &_graph,
                                        vector<pair<int,int>> &_vertex_pairs,
                                        vector<int> &_answer)
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
