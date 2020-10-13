#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _T>
void SCC::detect_trivial_components(VectCSRGraph &_graph,
                                    GraphAbstractionsNEC &_graph_API,
                                    FrontierNEC &_frontier,
                                    VerticesArrayNec<_T> &_forward_result,
                                    VerticesArrayNec<_T> &_backward_result,
                                    VerticesArrayNec<_T> &_trees,
                                    VerticesArrayNec<_T> &_active)
{
    _graph_API.change_traversal_direction(SCATTER);
    _graph_API.set_correct_direction(_forward_result, _frontier);
    auto forward_mark = [&_forward_result] (int src_id, int connections_count, int vector_index)
    {
        if(connections_count > 0)
            _forward_result[src_id] = 0;
        else
            _forward_result[src_id] = 1;
    };
    _frontier.set_all_active();
    _graph_API.compute(_graph, _frontier, forward_mark);

    _graph_API.change_traversal_direction(GATHER);
    _graph_API.set_correct_direction(_backward_result, _frontier);
    auto backward_mark = [&_backward_result] (int src_id, int connections_count, int vector_index)
    {
        if(connections_count > 0)
            _backward_result[src_id] = 0;
        else
            _backward_result[src_id] = 1;
    };

    _graph_API.compute(_graph, _frontier, backward_mark);

    _graph_API.change_traversal_direction(SCATTER);
    _graph.reorder(_backward_result, SCATTER);
    _graph_API.set_correct_direction(_frontier);
    int vertices_count = _graph.get_vertices_count();
    auto remove_trivial_and_init = [&_forward_result, &_backward_result, &_trees, &_active, vertices_count] (int src_id, int connections_count, int vector_index)
    {
        if((_forward_result[src_id] == 1) || (_backward_result[src_id] == 1))
        {
            _trees[src_id] = src_id + vertices_count;
            _active[src_id] = IS_NOT_ACTIVE;
        }
        else
        {
            _trees[src_id] = INIT_TREE;
            _active[src_id] = IS_ACTIVE;
        }
    };
    _frontier.set_all_active();
    _graph_API.compute(_graph, _frontier, remove_trivial_and_init);
    cout << "init done" << endl;
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _T>
void SCC::bfs_reach(VectCSRGraph &_graph,
                    GraphAbstractionsNEC &_graph_API,
                    FrontierNEC &_frontier,
                    VerticesArrayNec<_T> &_bfs_result,
                    int _source_vertex,
                    TraversalDirection _traversal_direction)
{
    _graph_API.change_traversal_direction(_traversal_direction);
    _graph_API.set_correct_direction(_bfs_result, _frontier);

    _frontier.set_all_active();

    auto init_levels = [&_bfs_result, _source_vertex] (int src_id, int connections_count, int vector_index)
    {
        if(src_id == _source_vertex)
            _bfs_result[_source_vertex] = FIRST_LEVEL_VERTEX;
        else
            _bfs_result[src_id] = UNVISITED_VERTEX;
    };
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

        auto on_next_level = [&_bfs_result, current_level] (int src_id)->int
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
                      VerticesArrayNec<_T> &_trees,
                      int _tree_num)
{
    _graph_API.change_traversal_direction(SCATTER);
    _graph_API.set_correct_direction(_frontier);

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
                         VerticesArrayNec<_T> &_forward_result,
                         VerticesArrayNec<_T> &_backward_result,
                         VerticesArrayNec<_T> &_trees,
                         VerticesArrayNec<_T> &_active,
                         int _last_tree)
{
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
                  VerticesArrayNec<_T> &_trees,
                  VerticesArrayNec<_T> &_forward_result,
                  VerticesArrayNec<_T> &_backward_result,
                  VerticesArrayNec<_T> &_active,
                  int _processed_tree,
                  int &_last_tree)
{
    int scatter_pivot = select_pivot(_graph, _graph_API, _frontier, _trees, _processed_tree);
    if(scatter_pivot == ERROR_IN_PIVOT)
        return;

    int gather_pivot = _graph.reorder(scatter_pivot, SCATTER, GATHER);

    bfs_reach(_graph, _graph_API, _frontier, _forward_result, scatter_pivot, SCATTER);
    bfs_reach(_graph, _graph_API, _frontier, _backward_result, gather_pivot, GATHER);

    _graph.reorder(_backward_result, SCATTER);

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
void SCC::nec_forward_backward(VectCSRGraph &_graph, VerticesArrayNec<_T> &_components)
{
    GraphAbstractionsNEC graph_API(_graph, SCATTER);
    FrontierNEC frontier(_graph, SCATTER);

    VerticesArrayNec<_T> forward_result(_graph, SCATTER);
    VerticesArrayNec<_T> backward_result(_graph, GATHER);
    VerticesArrayNec<_T> active(_graph, SCATTER);

    Timer tm, tm_init;
    tm.start();
    tm_init.start();
    detect_trivial_components(_graph, graph_API, frontier, forward_result, backward_result, _components, active);
    tm_init.end();
    cout << "init time: " << tm_init.get_time() << " sec" << endl;

    int last_tree = INIT_TREE;
    FB_step(_graph, graph_API, frontier, _components, forward_result, backward_result, active, INIT_TREE, last_tree);
    tm.end();
    cout << "last tree: " << last_tree << endl;

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    PerformanceStats::print_performance_stats("SCC (Forward-Backward)", tm.get_time(), _graph.get_edges_count());
    #endif
    print_component_sizes(_graph, _components);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

