#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
template <typename _T>
void RW::vgl_random_walk(VGL_Graph &_graph,
                         vector<int> &_walk_vertices,
                         int _walk_vertices_num,
                         int _walk_lengths,
                         VerticesArray<_T> &_walk_results)
{
    RandomGenerator rng;
    int *rand_array;
    MemoryAPI::allocate_array(&rand_array, _walk_vertices_num);

    VGL_GRAPH_ABSTRACTIONS graph_API(_graph);
    VGL_FRONTIER frontier(_graph, SCATTER);

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
    frontier.clear();
    frontier.add_group_of_vertices(&_walk_vertices[0], _walk_vertices.size());

    _walk_results.set_all_constant(DEAD_END);
    auto init_walks = [_walk_results] __VGL_COMPUTE_ARGS__ {
        _walk_results[src_id] = src_id;
    };
    graph_API.compute(_graph, frontier, init_walks);

    for(int iteration = 0; iteration < _walk_lengths; iteration++)
    {
        rng.generate_array_of_random_values(rand_array, _walk_vertices_num);

        auto visit_next = [iteration, _walk_lengths, rand_array, _walk_results, &_graph, walk_positions] __VGL_COMPUTE_ARGS__ {
            int walk_id = src_id;
            int current_id = _walk_results[walk_id];
            int current_connections_count = _graph.get_outgoing_connections_count(current_id);

            if((current_connections_count > 0) && (current_id != DEAD_END))
            {
                int walk_pos = walk_positions[walk_id];
                int rand_pos = rand_array[walk_pos] % current_connections_count;
                int next_vertex = _graph.get_outgoing_edge_dst(current_id, rand_pos);
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

    MemoryAPI::free_array(rand_array);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void RW::seq_random_walk(VGL_Graph &_graph,
                         vector<int> &_walk_vertices,
                         int _walk_vertices_num,
                         int _walk_lengths,
                         VerticesArray<_T> &_walk_results)
{
    // prepare walk positions
    Timer tm;
    tm.start();
    _walk_results.set_all_constant(DEAD_END);
    for(auto src_id: _walk_vertices)
    {
        _walk_results[src_id] = src_id;
    }

    for(int iteration = 0; iteration < _walk_lengths; iteration++)
    {
        int walk_pos = 0;
        for(auto src_id: _walk_vertices)
        {
            int walk_id = src_id;
            int current_id = _walk_results[walk_id];
            int connections_count = _graph.get_outgoing_connections_count(src_id);

            if((connections_count > 0) && (current_id != DEAD_END))
            {
                int rand_pos = rand() % connections_count;
                int next_vertex = _graph.get_edge_dst(current_id, rand_pos, SCATTER);
                _walk_results[walk_id] = next_vertex;
            }
            else
            {
                _walk_results[walk_id] = DEAD_END;
            }
            walk_pos++;
        }
    }
    tm.end();
    tm.print_time_stats("SEQ random walk");
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
