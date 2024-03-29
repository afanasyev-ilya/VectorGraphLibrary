#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void SCC::trim_step(VGL_Graph &_graph,
                    VGL_GRAPH_ABSTRACTIONS &_graph_API,
                    VGL_FRONTIER &_frontier,
                    VerticesArray<_T> &_out_degrees,
                    VerticesArray<_T> &_in_degrees,
                    VerticesArray<_T> &_trees,
                    VerticesArray<_T> &_active)
{
    VGL_FRONTIER out_frontier(_graph, SCATTER);
    VGL_FRONTIER in_frontier(_graph, GATHER);

    // set everythig as active
    _graph_API.change_traversal_direction(SCATTER, _frontier, _trees, _active);

    auto init = [_trees, _active] __VGL_COMPUTE_ARGS__
    {
        _trees[src_id] = INIT_TREE;
        _active[src_id] = IS_ACTIVE;
    };
    _frontier.set_all_active();
    _graph_API.compute(_graph, _frontier, init);

    int trim_steps = 0;
    int changes = 0;
    do
    {
        changes = 0;

        /* process out-degrees */
        _graph_API.change_traversal_direction(SCATTER, _frontier, out_frontier, _out_degrees, _active);

        // work only with low-degree vertices
        auto out_connections = [] __VGL_GNF_ARGS__
        {
            int result = NOT_IN_FRONTIER_FLAG;
            if(connections_count < VECTOR_LENGTH)
                result = IN_FRONTIER_FLAG;
            return result;
        };
        _graph_API.generate_new_frontier(_graph, out_frontier, out_connections);

        // init out-degrees
        _frontier.set_all_active();
        auto init_out_degrees = [_out_degrees] __VGL_COMPUTE_ARGS__
        {
            _out_degrees[src_id] = connections_count;
        };
        _graph_API.compute(_graph, _frontier, init_out_degrees);

        // if adjacnet is not active, decrese the degree
        auto update_out = [_active, _out_degrees] __VGL_SCATTER_ARGS__
        {
            if(_active[dst_id] == IS_NOT_ACTIVE)
                VGL_DEC(_out_degrees[src_id]); // _out_degrees[src_id] --
        };

        #ifdef __USE_NEC_SX_AURORA__
        _graph_API.enable_safe_stores();
        _graph_API.scatter(_graph, out_frontier, EMPTY_EDGE_OP, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP,
                           update_out, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);
        _graph_API.disable_safe_stores();
        #else
        _graph_API.scatter(_graph, out_frontier, update_out);
        #endif

        /* process in-degrees */
        // work only with low-degree vertices
        _graph_API.change_traversal_direction(GATHER, _frontier, in_frontier, _in_degrees, _active);

        auto in_connections = [] __VGL_GNF_ARGS__
        {
            int result = NOT_IN_FRONTIER_FLAG;
            if(connections_count < VECTOR_LENGTH)
                result = IN_FRONTIER_FLAG;
            return result;
        };
        _graph_API.generate_new_frontier(_graph, in_frontier, in_connections);

        // init out-degrees
        _frontier.set_all_active();
        auto init_in_degrees = [_in_degrees] __VGL_COMPUTE_ARGS__
        {
            _in_degrees[src_id] = connections_count;
        };
        _graph_API.compute(_graph, _frontier, init_in_degrees);

        // if adjacnet is not active, decrese the degree
        auto update_in = [_active, _in_degrees] __VGL_GATHER_ARGS__
        {
            if(_active[dst_id] == IS_NOT_ACTIVE)
                VGL_DEC(_in_degrees[src_id]);
        };
        #ifdef __USE_NEC_SX_AURORA__
        _graph_API.enable_safe_stores();
        _graph_API.gather(_graph, in_frontier, EMPTY_EDGE_OP, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP,
                          update_in, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);
        _graph_API.disable_safe_stores();
        #else
        _graph_API.gather(_graph, in_frontier, update_in);
        #endif

        _graph_API.change_traversal_direction(SCATTER, _frontier, _in_degrees, _out_degrees, _trees, _active);

        int vertices_count = _graph.get_vertices_count();
        VEC_REGISTER_INT(changes, 0);
        auto remove_trivial_and_init = [_active, _in_degrees, _out_degrees, _trees, VGL_LAMBDA_CAP(reg_changes), vertices_count] __VGL_COMPUTE_ARGS__
        {
            if((_active[src_id] == IS_ACTIVE) && ((_in_degrees[src_id] <= 0) || (_out_degrees[src_id] <= 0)))
            {
                _trees[src_id] = src_id + vertices_count;
                _active[src_id] = IS_NOT_ACTIVE;
                reg_changes[vector_index] = 1;
            }
        };
        _frontier.set_all_active();
        _graph_API.compute(_graph, _frontier, remove_trivial_and_init);
        changes = register_sum_reduce(reg_changes);
        register_free(reg_changes);
        trim_steps++;
    } while(changes);

    cout << "Did " << trim_steps << " trim steps" << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void SCC::bfs_reach(VGL_Graph &_graph,
                    VGL_GRAPH_ABSTRACTIONS  &_graph_API,
                    VGL_FRONTIER &_frontier,
                    VerticesArray<_T> &_bfs_result,
                    int _source_vertex,
                    TraversalDirection _traversal_direction)
{
    _graph_API.change_traversal_direction(_traversal_direction, _bfs_result, _frontier, _bfs_result);

    auto init_levels = [_bfs_result, _source_vertex] __VGL_COMPUTE_ARGS__
    {
        if(src_id == _source_vertex)
            _bfs_result[_source_vertex] = FIRST_LEVEL_VERTEX;
        else
            _bfs_result[src_id] = UNVISITED_VERTEX;
    };
    _frontier.set_all_active();
    _graph_API.compute(_graph, _frontier, init_levels);

    _frontier.clear();
    _frontier.add_vertex(_source_vertex);

    int current_level = FIRST_LEVEL_VERTEX;
    while(_frontier.size() > 0)
    {
        auto edge_op = [_bfs_result, current_level] __VGL_ADVANCE_ARGS__
        {
            int dst_level = _bfs_result[dst_id];
            if(dst_level == UNVISITED_VERTEX)
            {
                _bfs_result[dst_id] = current_level + 1;
            }
        };

        if(_traversal_direction == SCATTER)
            _graph_API.scatter(_graph, _frontier, edge_op);
        else if(_traversal_direction == GATHER)
            _graph_API.gather(_graph, _frontier, edge_op);

        auto on_next_level = [_bfs_result, current_level] __VGL_GNF_ARGS__
        {
            int result = NOT_IN_FRONTIER_FLAG;
            if(_bfs_result[src_id] == (current_level + 1))
                result = IN_FRONTIER_FLAG;
            return result;
        };

        _graph_API.generate_new_frontier(_graph, _frontier, on_next_level);

        current_level++;
    }

    if(_traversal_direction == GATHER) // change direction back to SCATTER if required
        _graph_API.change_traversal_direction(_traversal_direction);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
int SCC::select_pivot(VGL_Graph &_graph,
                      VGL_GRAPH_ABSTRACTIONS  &_graph_API,
                      VGL_FRONTIER &_frontier,
                      VerticesArray<_T> &_trees,
                      int _tree_num)
{
    _graph_API.change_traversal_direction(SCATTER, _frontier, _trees);

    VEC_REGISTER_INT(pivots, _graph.get_vertices_count() + 1);

    auto can_be_pivot = [_trees, _tree_num, VGL_LAMBDA_CAP(reg_pivots)] __VGL_COMPUTE_ARGS__
    {
        if(_trees[src_id] == _tree_num)
            reg_pivots[vector_index] = src_id;
    };
    _frontier.set_all_active();
    _graph_API.compute(_graph, _frontier, can_be_pivot);

    int pivot = _graph.get_vertices_count() + 1;
    for(int i = 0; i < VECTOR_LENGTH; i++)
    {
        if(reg_pivots[i] < pivot)
            pivot = reg_pivots[i];
    }

    if(pivot > _graph.get_vertices_count())
        pivot = ERROR_IN_PIVOT;

    register_free(reg_pivots);

    return pivot;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void SCC::process_result(VGL_Graph &_graph,
                         VGL_GRAPH_ABSTRACTIONS  &_graph_API,
                         VGL_FRONTIER &_frontier,
                         VerticesArray<_T> &_forward_result,
                         VerticesArray<_T> &_backward_result,
                         VerticesArray<_T> &_trees,
                         VerticesArray<_T> &_active,
                         int _last_tree)
{
    _graph_API.change_traversal_direction(SCATTER, _frontier, _forward_result, _backward_result, _trees, _active);

    auto locate_scc = [_forward_result, _backward_result, _trees, _active, _last_tree] __VGL_COMPUTE_ARGS__
    {
        _T fwd_res = _forward_result[src_id];
        _T bwd_res = _backward_result[src_id];
        _T active = _active[src_id];

        if ((active == IS_ACTIVE) && (fwd_res != UNVISITED_VERTEX) && (bwd_res != UNVISITED_VERTEX))
        {
            _trees[src_id] = _last_tree;
            _active[src_id] = IS_NOT_ACTIVE;
        }
        if ((active == IS_ACTIVE) && (fwd_res == UNVISITED_VERTEX) && (bwd_res != UNVISITED_VERTEX))
        {
            _trees[src_id] = _last_tree + 1;
        }
        if ((active == IS_ACTIVE) && (fwd_res != UNVISITED_VERTEX) && (bwd_res == UNVISITED_VERTEX))
        {
            _trees[src_id] = _last_tree + 2;
        }
        if ((active == IS_ACTIVE) && (fwd_res == UNVISITED_VERTEX) && (bwd_res == UNVISITED_VERTEX))
        {
            _trees[src_id] = _last_tree + 3;
        }
    };

    _frontier.set_all_active();
    _graph_API.compute(_graph, _frontier, locate_scc);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void SCC::FB_step(VGL_Graph &_graph,
                  VGL_GRAPH_ABSTRACTIONS &_graph_API,
                  VGL_FRONTIER &_frontier,
                  VerticesArray<_T> &_trees,
                  VerticesArray<_T> &_forward_result,
                  VerticesArray<_T> &_backward_result,
                  VerticesArray<_T> &_active,
                  int _processed_tree,
                  int &_last_tree)
{
    int scatter_pivot = select_pivot(_graph, _graph_API, _frontier, _trees, _processed_tree);
    if(scatter_pivot == ERROR_IN_PIVOT)
        return;
    int gather_pivot = _graph.reorder(scatter_pivot, SCATTER, GATHER);

    /*cout << "non-trivial FB step with pivot " << scatter_pivot << endl;
    _graph.print_vertex_information(SCATTER, scatter_pivot, 20);
    _graph.print_vertex_information(GATHER, gather_pivot, 20);
    cout << " --------------- " << endl;*/

    bfs_reach(_graph, _graph_API, _frontier, _forward_result, scatter_pivot, SCATTER);
    bfs_reach(_graph, _graph_API, _frontier, _backward_result, gather_pivot, GATHER);

    int current_tree = _last_tree;
    process_result(_graph, _graph_API, _frontier, _forward_result, _backward_result, _trees, _active, _last_tree);
    _last_tree += 4;

    FB_step(_graph, _graph_API, _frontier, _trees, _forward_result, _backward_result, _active, current_tree + 1, _last_tree);
    FB_step(_graph, _graph_API, _frontier, _trees, _forward_result, _backward_result, _active, current_tree + 2, _last_tree);
    FB_step(_graph, _graph_API, _frontier, _trees, _forward_result, _backward_result, _active, current_tree + 3, _last_tree);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
double SCC::vgl_forward_backward(VGL_Graph &_graph, VerticesArray<_T> &_components)
{
    VGL_GRAPH_ABSTRACTIONS graph_API(_graph, SCATTER);
    VGL_FRONTIER frontier(_graph, SCATTER);

    VerticesArray<_T> forward_result(_graph, SCATTER);
    VerticesArray<_T> backward_result(_graph, GATHER);
    VerticesArray<_T> active(_graph, SCATTER);

    Timer trim_tm;
    trim_tm.start();
    trim_step(_graph, graph_API, frontier, forward_result, backward_result, _components, active);
    trim_tm.end();

    int last_tree = INIT_TREE;
    Timer bfs_tm;
    bfs_tm.start();
    FB_step(_graph, graph_API, frontier, _components, forward_result, backward_result, active, INIT_TREE, last_tree);
    bfs_tm.end();

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    cout << "last tree: " << last_tree << endl;
    cout << "trim time: " << trim_tm.get_time_in_ms() << " ms" << endl;
    cout << "bfs time:" << bfs_tm.get_time_in_ms() << " ms" << endl;
    performance_stats.print_algorithm_performance_stats("SCC (Forward-Backward)", trim_tm.get_time() + bfs_tm.get_time(), _graph.get_edges_count());
    print_component_sizes(_components);
    #endif

    return performance_stats.get_algorithm_performance(trim_tm.get_time() + bfs_tm.get_time(), _graph.get_edges_count());
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

