#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double TransitiveClosure::vgl_purdom(VGL_Graph &_graph,
                                     vector<pair<int,int>> &_vertex_pairs,
                                     vector<bool> &_answer)
{
    // compute SCC for original graph
    VerticesArray<int> components(_graph);
    SCC::vgl_forward_backward(_graph, components);

    components.print();

    // compute intermediate representation graph
    EdgesArray<int> new_src_ids(_graph);
    EdgesArray<int> new_dst_ids(_graph);

    VGL_GRAPH_ABSTRACTIONS graph_API(_graph);
    VGL_FRONTIER frontier(_graph);
    graph_API.change_traversal_direction(SCATTER, frontier, components);

    frontier.set_all_active();
    auto edge_op = [components, new_src_ids, new_dst_ids] __VGL_SCATTER_ARGS__ {
        int src_comp = components[src_id];
        int dst_comp = components[dst_id];
        if(src_comp == dst_comp)
        {
            new_src_ids[global_edge_pos] = src_comp;
            new_dst_ids[global_edge_pos] = src_comp;
        }
        else
        {
            new_src_ids[global_edge_pos] = -1;
            new_dst_ids[global_edge_pos] = -1;
        }
    };
    graph_API.scatter(_graph, frontier, edge_op);
    new_src_ids.print();
    new_dst_ids.print();

    int *edges_buffer;
    int *edge_indexes;
    long long edges_count = _graph.get_edges_count();
    MemoryAPI::allocate_array(&edges_buffer, edges_count);
    MemoryAPI::allocate_array(&edge_indexes, edges_count);

    auto different_components = [new_src_ids, new_dst_ids] (int idx) {
        if(new_src_ids[idx] != -1 && new_dst_ids[idx] != -1)
            return 1;
        else
            return 0;
    };
    long long new_edges_count = ParallelPrimitives::copy_if_indexes(different_components,
          edge_indexes, edges_count, edges_buffer, edges_count, 0);

    MemoryAPI::free_array(edges_buffer);

    auto max_component_num = [components]__VGL_REDUCE_ANY_ARGS__->int
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

    tmp_edges.print();

    // merge if required

    // do copy if from edges arrays

    // init new graph

    //EdgesContainer tmp_edges;
    //tmp_edges
    //MemoryAPI::allocate_array();
    // filter edges
   // VGL_Graph intermediate_representation_graph(EDGES_LIST_GRAPH);
    //intermediate_representation_graph.import(tmp_edges);

    // do BFS

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
