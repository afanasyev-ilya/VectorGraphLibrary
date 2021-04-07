#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void RW::vgl_random_walk(VectCSRGraph &_graph,
                         std::set<int> &_walk_vertices,
                         int _walk_vertices_num,
                         int _walk_lengths,
                         VerticesArray<_T> &_walk_results,
                         int *_walk_paths)
{
    int store_walk_paths = false;
    if(_walk_paths != NULL)
        store_walk_paths = true;

    VGL_GRAPH_ABSTRACTIONS graph_API(_graph);
    VGL_FRONTIER frontier(_graph);

    // prepare walk positions
    Timer tm;
    tm.start();
    int position = 0;
    VerticesArray<int> walk_positions(_graph);
    for (const int &vertex : _walk_vertices)
    {
        walk_positions[vertex] = position;
        position++;
    }
    tm.end();
    tm.print_time_stats("prepare walk positions");

    tm.start();
    auto is_walk_vertex = [&_walk_results, &_walk_vertices] __VGL_GNF_ARGS__
    {
        int result = NOT_IN_FRONTIER_FLAG;
        if(_walk_vertices.count(src_id) > 0)
            result = IN_FRONTIER_FLAG;
        return result;
    };
    graph_API.generate_new_frontier(_graph, frontier, is_walk_vertex);

    auto init_walks = [&_walk_results] __VGL_COMPUTE_ARGS__
    {
        _walk_results[src_id] = src_id;
    };
    _walk_results.set_all_constant(DEAD_END);
    graph_API.compute(_graph, frontier, init_walks);

    for(int iteration = 0; iteration < _walk_lengths; iteration++)
    {
        auto visit_next = [iteration, store_walk_paths, _walk_lengths, &_walk_results, &_graph, &walk_positions, &_walk_paths] __VGL_COMPUTE_ARGS__
        {
            int walk_id = src_id;
            int current_id = _walk_results[walk_id];

            if((connections_count > 0) && (current_id != DEAD_END))
            {
                int next_vertex_pos = rand() % connections_count;
                int next_vertex = _graph.get_edge_dst(current_id, next_vertex_pos, SCATTER);

                if(store_walk_paths)
                {
                    int walk_pos = walk_positions[walk_id];
                    long long dst_index = walk_pos * _walk_lengths + iteration;
                    _walk_paths[dst_index] = next_vertex;
                }

                _walk_results[walk_id] = next_vertex;
            }
            else
            {
                _walk_results[walk_id] = DEAD_END;
            }
        };

        graph_API.compute(_graph, frontier, visit_next);
    }
    tm.end();
    tm.print_time_stats("RW algorithm");
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
