#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void HITS::vgl_hits(VectCSRGraph &_graph, VerticesArray<_T> &_auth, VerticesArray<_T> &_hub, int _num_steps)
{
    VGL_GRAPH_ABSTRACTIONS graph_API(_graph);
    VGL_FRONTIER frontier(_graph);
    int vertices_count = _graph.get_vertices_count();

    frontier.set_all_active();

    Timer tm;
    tm.start();

    auto init_op = [_auth, _hub] __VGL_COMPUTE_ARGS__ {
        _auth[src_id] = 1;
        _hub[src_id] = 1;
    };
    graph_API.compute(_graph, frontier, init_op);

    for(int step = 0; step < _num_steps; step++)
    {
        graph_API.change_traversal_direction(GATHER, _hub, _auth, frontier);

        auto update_auth_op_preprocess = [_auth] __VGL_ADVANCE_PREPROCESS_ARGS__ {
            _auth[src_id] = 0.0;
        };

        auto update_auth_op = [_auth, _hub, vertices_count] __VGL_ADVANCE_ARGS__ {
            _auth[src_id] += _hub[dst_id];
        };
        graph_API.gather(_graph, frontier, update_auth_op, update_auth_op_preprocess, EMPTY_VERTEX_OP,
                         update_auth_op, update_auth_op_preprocess, EMPTY_VERTEX_OP);

        auto reduce_auth_op = [_auth] __VGL_REDUCE_DBL_ARGS__ {
            return _auth[src_id] * _auth[src_id];
        };
        _T norm = sqrt(graph_API.reduce<_T>(_graph, frontier, reduce_auth_op, REDUCE_SUM));

        auto normalize_auth_op = [_auth, norm] __VGL_COMPUTE_ARGS__ {
            _auth[src_id] /= norm;
        };
        graph_API.compute(_graph, frontier, normalize_auth_op);

        graph_API.change_traversal_direction(SCATTER, _hub, _auth, frontier);

        auto update_hub_op_preprocess = [_hub] __VGL_ADVANCE_PREPROCESS_ARGS__ {
            _hub[src_id] = 0.0;
        };

        auto update_hub_op = [_hub, _auth] __VGL_ADVANCE_ARGS__ {
            _hub[src_id] += _auth[dst_id];
        };
        graph_API.scatter(_graph, frontier, update_hub_op, update_hub_op_preprocess, EMPTY_VERTEX_OP,
                          update_hub_op, update_hub_op_preprocess, EMPTY_VERTEX_OP);

        auto reduce_hub_op = [_hub] __VGL_REDUCE_DBL_ARGS__ {
            return _hub[src_id] * _hub[src_id];
        };
        norm = sqrt(graph_API.reduce<_T>(_graph, frontier, reduce_hub_op, REDUCE_SUM));

        auto normalize_hub_op = [_hub, norm] __VGL_COMPUTE_ARGS__ {
                _hub[src_id] /= norm;
        };
        graph_API.compute(_graph, frontier, normalize_hub_op);
    }
    tm.end();

    performance_stats.save_algorithm_performance_stats(tm.get_time(), _graph.get_edges_count(), _num_steps);
    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    performance_stats.print_algorithm_performance_stats("VGL HITS");
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void HITS::seq_hits(VectCSRGraph &_graph, VerticesArray<_T> &_auth, VerticesArray<_T> &_hub, int _num_steps)
{
    Timer tm;
    tm.start();

    int vertices_count = _graph.get_vertices_count();
    for(int src_id = 0; src_id < vertices_count; src_id++)
    {
        _auth[src_id] = 1;
        _hub[src_id] = 1;
    }

    for(int step = 0; step < _num_steps; step++)
    {
        _auth.reorder(GATHER);
        _hub.reorder(GATHER);

        _T norm = 0.0;
        for(int src_id = 0; src_id < vertices_count; src_id++)
        {
            _T p_auth = 0.0;

            for(int i = 0; i < _graph.get_incoming_connections_count(src_id); i++)
            {
                int dst_id = _graph.get_incoming_edge_dst(src_id, i);
                p_auth += _hub[dst_id];
            }

            _auth[src_id] = p_auth;
            norm += p_auth * p_auth;
        }

        norm = sqrt(norm);

        for(int src_id = 0; src_id < vertices_count; src_id++)
        {
            _auth[src_id] /= norm;
        }

        _auth.reorder(SCATTER);
        _hub.reorder(SCATTER);

        norm = 0.0;
        for(int src_id = 0; src_id < vertices_count; src_id++)
        {
            _T p_hub = 0.0;
            for(int i = 0; i < _graph.get_outgoing_connections_count(src_id); i++)
            {
                int dst_id = _graph.get_outgoing_edge_dst(src_id, i);
                p_hub += _auth[dst_id];
            }
            _hub[src_id] = p_hub;
            norm += p_hub * p_hub;
        }

        norm = sqrt(norm);

        for(int src_id = 0; src_id < vertices_count; src_id++)
        {
            _hub[src_id] /= norm;
        }
    }
    tm.end();

    performance_stats.save_algorithm_performance_stats(tm.get_time(), _graph.get_edges_count(), _num_steps);
    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    performance_stats.print_algorithm_performance_stats("SEQ HITS");
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
