#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _T>
void SCC::trim_step(VectCSRGraph &_graph,
                    GraphAbstractionsNEC &_graph_API,
                    FrontierNEC &_frontier,
                    VerticesArrayNEC<_T> &_out_degrees,
                    VerticesArrayNEC<_T> &_in_degrees,
                    VerticesArrayNEC<_T> &_trees,
                    VerticesArrayNEC<_T> &_active)
{
    FrontierNEC out_frontier(_graph, SCATTER);
    FrontierNEC in_frontier(_graph, GATHER);

    // set everythig as active
    _graph_API.change_traversal_direction(SCATTER, _frontier, _trees, _active);

    auto init = [&_trees, &_active] (int src_id, int connections_count, int vector_index)
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
        auto out_connections = [] (int src_id, int connections_count)->int
        {
            int result = NOT_IN_FRONTIER_FLAG;
            if(connections_count < VECTOR_LENGTH)
                result = IN_FRONTIER_FLAG;
            return result;
        };
        _graph_API.generate_new_frontier(_graph, out_frontier, out_connections);

        // init out-degrees
        _frontier.set_all_active();
        auto init_out_degrees = [&_out_degrees] (int src_id, int connections_count, int vector_index)
        {
            _out_degrees[src_id] = connections_count;
        };
        _graph_API.compute(_graph, _frontier, init_out_degrees);

        // if adjacnet is not active, decrese the degree
        auto update_out = [&_active, &_out_degrees](int src_id, int dst_id, int local_edge_pos,
                long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
        {
            if(_active[dst_id] == IS_NOT_ACTIVE)
                _out_degrees[src_id]--;
        };
        _graph_API.scatter(_graph, out_frontier, update_out, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP,
                           update_out, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);

        /* process in-degrees */
        // work only with low-degree vertices
        _graph_API.change_traversal_direction(GATHER, _frontier, in_frontier, _in_degrees, _active);

        auto in_connections = [] (int src_id, int connections_count)->int
        {
            int result = NOT_IN_FRONTIER_FLAG;
            if(connections_count < VECTOR_LENGTH)
                result = IN_FRONTIER_FLAG;
            return result;
        };
        _graph_API.generate_new_frontier(_graph, in_frontier, in_connections);

        // init out-degrees
        _frontier.set_all_active();
        auto init_in_degrees = [&_in_degrees] (int src_id, int connections_count, int vector_index)
        {
            _in_degrees[src_id] = connections_count;
        };
        _graph_API.compute(_graph, _frontier, init_in_degrees);

        // if adjacnet is not active, decrese the degree
        auto update_in = [&_active, &_in_degrees](int src_id, int dst_id, int local_edge_pos,
                long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
        {
            if(_active[dst_id] == IS_NOT_ACTIVE)
                _in_degrees[src_id]--;
        };
        _graph_API.gather(_graph, in_frontier, update_in, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP,
                          update_in, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);

        _graph_API.change_traversal_direction(SCATTER, _frontier, _in_degrees, _out_degrees, _trees, _active);

        int vertices_count = _graph.get_vertices_count();
        NEC_REGISTER_INT(changes, 0);
        auto remove_trivial_and_init = [&_active, &_in_degrees, &_out_degrees, &_trees, &reg_changes, vertices_count] (int src_id, int connections_count, int vector_index)
        {
            if((_active[src_id] == IS_ACTIVE) && ((_in_degrees[src_id] == 0) || (_out_degrees[src_id] == 0)))
            {
                _trees[src_id] = src_id + vertices_count;
                _active[src_id] = IS_NOT_ACTIVE;
                reg_changes[vector_index] = 1;
            }
        };
        _frontier.set_all_active();
        _graph_API.compute(_graph, _frontier, remove_trivial_and_init);
        changes = register_sum_reduce(reg_changes);
        trim_steps++;
    } while(changes);

    cout << "Did " << trim_steps << " trim steps" << endl;
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _T>
void SCC::bfs_reach(VectCSRGraph &_graph,
                    GraphAbstractionsNEC &_graph_API,
                    FrontierNEC &_frontier,
                    VerticesArrayNEC<_T> &_bfs_result,
                    int _source_vertex,
                    TraversalDirection _traversal_direction)
{
    _graph_API.change_traversal_direction(_traversal_direction, _bfs_result, _frontier, _bfs_result);

    auto init_levels = [&_bfs_result, _source_vertex] (int src_id, int connections_count, int vector_index)
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
        auto edge_op = [&_bfs_result, current_level](int src_id, int dst_id, int local_edge_pos,
                long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
        {
            int dst_level = _bfs_result[dst_id];
            if(dst_level == UNVISITED_VERTEX)
            {
                _bfs_result[dst_id] = current_level + 1;
            }
        };

        if(_traversal_direction == SCATTER)
            _graph_API.scatter(_graph, _frontier, edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP,
                               edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);
        else if(_traversal_direction == GATHER)
            _graph_API.gather(_graph, _frontier, edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP,
                              edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);

        auto on_next_level = [&_bfs_result, current_level] (int src_id, int connections_count)->int
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
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _T>
int SCC::select_pivot(VectCSRGraph &_graph,
                      GraphAbstractionsNEC &_graph_API,
                      FrontierNEC &_frontier,
                      VerticesArrayNEC<_T> &_trees,
                      int _tree_num)
{
    _graph_API.change_traversal_direction(SCATTER, _frontier, _trees);

    NEC_REGISTER_INT(pivots, _graph.get_vertices_count() + 1);

    auto select_pivot = [&_trees, _tree_num, &reg_pivots] (int src_id, int connections_count, int vector_index)
    {
        if(_trees[src_id] == _tree_num)
            reg_pivots[vector_index] = src_id;
    };
    _frontier.set_all_active();
    _graph_API.compute(_graph, _frontier, select_pivot);

    int pivot = _graph.get_vertices_count() + 1;
    #pragma _NEC vector
    for(int i = 0; i < VECTOR_LENGTH; i++)
    {
        if(reg_pivots[i] < pivot)
            pivot = reg_pivots[i];
    }

    if(pivot > _graph.get_vertices_count())
        pivot = ERROR_IN_PIVOT;

    return pivot;
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _T>
void SCC::process_result(VectCSRGraph &_graph,
                         GraphAbstractionsNEC &_graph_API,
                         FrontierNEC &_frontier,
                         VerticesArrayNEC<_T> &_forward_result,
                         VerticesArrayNEC<_T> &_backward_result,
                         VerticesArrayNEC<_T> &_trees,
                         VerticesArrayNEC<_T> &_active,
                         int _last_tree)
{
    _graph_API.change_traversal_direction(SCATTER, _frontier, _forward_result, _backward_result, _trees, _active);

    auto locate_scc = [&_forward_result, &_backward_result, &_trees, &_active, _last_tree] (int src_id, int connections_count, int vector_index)
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
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _T>
void SCC::FB_step(VectCSRGraph &_graph,
                  GraphAbstractionsNEC &_graph_API,
                  FrontierNEC &_frontier,
                  VerticesArrayNEC<_T> &_trees,
                  VerticesArrayNEC<_T> &_forward_result,
                  VerticesArrayNEC<_T> &_backward_result,
                  VerticesArrayNEC<_T> &_active,
                  int _processed_tree,
                  int &_last_tree)
{
    int scatter_pivot = select_pivot(_graph, _graph_API, _frontier, _trees, _processed_tree);
    if(scatter_pivot == ERROR_IN_PIVOT)
        return;
    int gather_pivot = _graph.reorder(scatter_pivot, SCATTER, GATHER);

    bfs_reach(_graph, _graph_API, _frontier, _forward_result, scatter_pivot, SCATTER);
    bfs_reach(_graph, _graph_API, _frontier, _backward_result, gather_pivot, GATHER);

    int current_tree = _last_tree;
    process_result(_graph, _graph_API, _frontier, _forward_result, _backward_result, _trees, _active, _last_tree);
    _last_tree += 4;

    FB_step(_graph, _graph_API, _frontier, _trees, _forward_result, _backward_result, _active, current_tree + 1, _last_tree);
    FB_step(_graph, _graph_API, _frontier, _trees, _forward_result, _backward_result, _active, current_tree + 2, _last_tree);
    FB_step(_graph, _graph_API, _frontier, _trees, _forward_result, _backward_result, _active, current_tree + 3, _last_tree);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _T>
void SCC::nec_forward_backward(VectCSRGraph &_graph, VerticesArrayNEC<_T> &_components)
{
    GraphAbstractionsNEC graph_API(_graph, SCATTER);
    FrontierNEC frontier(_graph, SCATTER);

    VerticesArrayNEC<_T> forward_result(_graph, SCATTER);
    VerticesArrayNEC<_T> backward_result(_graph, GATHER);
    VerticesArrayNEC<_T> active(_graph, SCATTER);

    Timer tm;
    tm.start();
    trim_step(_graph, graph_API, frontier, forward_result, backward_result, _components, active);

    int last_tree = INIT_TREE;
    FB_step(_graph, graph_API, frontier, _components, forward_result, backward_result, active, INIT_TREE, last_tree);
    tm.end();
    cout << "last tree: " << last_tree << endl;

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    PerformanceStats::print_algorithm_performance_stats("SCC (Forward-Backward)", tm.get_time(), _graph.get_edges_count());
    #endif
    print_component_sizes(_graph, _components);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

