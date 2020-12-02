#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
void BFS::nec_top_down_compute_step(VectCSRGraph &_graph,
                                    VerticesArray<int> &_levels,
                                    int _current_level,
                                    int &_vis,
                                    int &_in_lvl,
                                    bool _compute_stats)
{
    /*if(_compute_stats)
    {
        #pragma omp parallel
        {
            NEC_REGISTER_INT(vis, 0);
            NEC_REGISTER_INT(in_lvl, 0);

            auto edge_op_with_stats = [_levels, _current_level, &reg_vis, &reg_in_lvl](int src_id, int dst_id, int local_edge_pos,
                    long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
            {
                int src_level = _levels[src_id];
                int dst_level = _levels[dst_id];
                reg_in_lvl[vector_index]++;
                if((src_level == _current_level) && (dst_level == UNVISITED_VERTEX))
                {
                    _levels[dst_id] = _current_level + 1;
                    reg_vis[vector_index]++;
                }
            };

            graph_API.advance(_graph, frontier, edge_op_with_stats);

            int local_vis = register_sum_reduce(reg_vis);
            int local_in_lvl = register_sum_reduce(reg_in_lvl);

            #pragma omp atomic
            _vis += local_vis;

            #pragma omp atomic
            _in_lvl += local_in_lvl;
        }
    }
    else
    {
        auto edge_op = [_levels, _current_level](int src_id, int dst_id, int local_edge_pos,
                long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
        {
            int src_level = _levels[src_id];
            int dst_level = _levels[dst_id];
            if((src_level == _current_level) && (dst_level == UNVISITED_VERTEX))
            {
                _levels[dst_id] = _current_level + 1;
            }
        };

        graph_API.advance(_graph, frontier, edge_op);
    }*/
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
void BFS::nec_bottom_up_compute_step(VectCSRGraph &_graph,
                                     VerticesArray<int> &_levels,
                                     int *_connections_array,
                                     int _current_level,
                                     int &_vis,
                                     int &_in_lvl,
                                     bool _use_vector_extension)
{
    /*int first_edge = 0;
    if(_use_vector_extension)
    {
        frontier.set_all_active();

        first_edge = 4;
        #pragma omp parallel
        {
            NEC_REGISTER_INT(vis, 0);
            NEC_REGISTER_INT(in_lvl, 0);

            auto edge_op = [_levels, _current_level, &reg_vis, &reg_in_lvl](int src_id, int dst_id, int local_edge_pos,
                    long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
            {
                if((_levels[src_id] == UNVISITED_VERTEX) && (_levels[dst_id] == _current_level))
                {
                    _levels[src_id] = _current_level + 1;
                    reg_vis[vector_index]++;
                }
            };

            graph_API.partial_advance(_graph, frontier, edge_op, 0, first_edge);

            int local_vis = register_sum_reduce(reg_vis);

            #pragma omp atomic
            _vis += local_vis;
        }
    }

    auto vertex_value_is_unset = [_levels, _connections_array, first_edge] (int src_id)->int
    {
        int result = NOT_IN_FRONTIER_FLAG;
        if((_levels[src_id] == UNVISITED_VERTEX) && (_connections_array[src_id] > 0))
            result = IN_FRONTIER_FLAG;
        return result;
    };
    graph_API.generate_new_frontier(_graph, frontier, vertex_value_is_unset); // TODO replace with filter

    #pragma omp parallel
    {
        NEC_REGISTER_INT(vis, 0);
        NEC_REGISTER_INT(in_lvl, 0);

        auto edge_op = [_levels, _current_level, &reg_vis, &reg_in_lvl](int src_id, int dst_id, int local_edge_pos,
                long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
        {
            reg_in_lvl[vector_index]++;
            if((_levels[src_id] == UNVISITED_VERTEX) && (_levels[dst_id] == _current_level))
            {
                _levels[src_id] = _current_level + 1;
                reg_vis[vector_index]++;
            }
        };

        auto edge_collective_op = [_levels, _current_level, &reg_vis, &reg_in_lvl](int src_id, int dst_id, int local_edge_pos,
                long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
        {
            reg_in_lvl[vector_index]++;
            if((_levels[src_id] == UNVISITED_VERTEX) && (_levels[dst_id] == _current_level))
            {
                _levels[src_id] = _current_level + 1;
                reg_vis[vector_index]++;
            }
        };

        graph_API.advance(_graph, frontier, edge_op);

        int local_vis = register_sum_reduce(reg_vis);
        int local_in_lvl = register_sum_reduce(reg_in_lvl);

        #pragma omp atomic
        _vis += local_vis;

        #pragma omp atomic
        _in_lvl += local_in_lvl;
    }*/
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*#ifdef __USE_NEC_SX_AURORA__
double BFS::nec_direction_optimizing(VectCSRGraph &_graph,
                                     int *_levels,
                                     int _source_vertex)
{
    LOAD_UNDIRECTED_CSR_GRAPH_DATA(_graph);
    GraphStructure graph_structure = check_graph_structure(_graph);
    int *connections_array;
    MemoryAPI::allocate_array(&connections_array, vertices_count);
    frontier.set_all_active();

    auto calculate_non_zero_count = [](int src_id, int connections_count, int vector_index)->int
    {
        int result = 0;
        if(connections_count > 0)
        {
            result = 1;
        }
        return result;
    };
    int non_zero_count = graph_API.reduce<int>(_graph, frontier, calculate_non_zero_count, REDUCE_SUM);

    auto set_reminder = [_levels, non_zero_count](int src_id, int connections_count, int vector_index)
    {
        if(src_id >= non_zero_count)
            _levels[src_id] = UNVISITED_VERTEX;
    };
    graph_API.compute(_graph, frontier, set_reminder);

    frontier.change_size(non_zero_count);
    frontier.set_all_active();

    auto init_connections = [connections_array] (int src_id, int connections_count, int vector_index)
    {
        connections_array[src_id] = connections_count;
    };
    graph_API.compute(_graph, frontier, init_connections);

    double t1 = omp_get_wtime();

    auto init_levels = [_levels, _source_vertex] (int src_id, int connections_count, int vector_index)
    {
        if(src_id == _source_vertex)
            _levels[_source_vertex] = FIRST_LEVEL_VERTEX;
        else
            _levels[src_id] = UNVISITED_VERTEX;
    };
    graph_API.compute(_graph, frontier, init_levels);

    frontier.clear();
    frontier.add_vertex(_graph, _source_vertex);

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    vector<double> step_times;
    vector<StateOfBFS> step_states;
    #endif

    int vis = 1, in_lvl = 0;
    int current_level = FIRST_LEVEL_VERTEX;
    StateOfBFS current_state = TOP_DOWN;
    bool _use_vect_CSR_extension = false;
    int current_frontier_size = 1, prev_frontier_size = 0;
    while(vis > 0)
    {
        #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
        step_states.push_back(current_state);
        cout << "step " << current_level - FIRST_LEVEL_VERTEX << " in state " << current_state << endl;
        #endif

        double t_st, t_end;

        vis = 0, in_lvl = 0;
        if(current_state == TOP_DOWN)
        {
            auto on_current_level = [_levels, current_level] (int src_id)->int
            {
                int result = NOT_IN_FRONTIER_FLAG;
                if(_levels[src_id] == current_level)
                    result = IN_FRONTIER_FLAG;
                return result;
            };

            #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
            t_st = omp_get_wtime();
            #endif

            NEC_REGISTER_INT(vis, 0);
            NEC_REGISTER_INT(in_lvl, 0);

            auto edge_op_with_stats = [_levels, current_level, &reg_vis, &reg_in_lvl](int src_id, int dst_id, int local_edge_pos,
                    long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
            {
                int dst_level = _levels[dst_id];
                reg_in_lvl[vector_index]++;
                if(dst_level == UNVISITED_VERTEX)
                {
                    _levels[dst_id] = current_level + 1;
                    reg_vis[vector_index]++;
                }
            };

            auto on_next_level = [_levels, current_level] (int src_id)->int
            {
                int result = NOT_IN_FRONTIER_FLAG;
                if(_levels[src_id] == (current_level + 1))
                    result = IN_FRONTIER_FLAG;
                return result;
            };

            if(current_level != FIRST_LEVEL_VERTEX)
                graph_API.advance(_graph, frontier, frontier, edge_op_with_stats, on_next_level);
            else
                graph_API.advance(_graph, frontier, edge_op_with_stats);

            int local_vis = register_sum_reduce(reg_vis);
            int local_in_lvl = register_sum_reduce(reg_in_lvl);

            #pragma omp atomic
            vis += local_vis;

            #pragma omp atomic
            in_lvl += local_in_lvl;

            #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
            t_end = omp_get_wtime();
            #endif
        }
        else if(current_state == BOTTOM_UP)
        {
            #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
            t_st = omp_get_wtime();
            #endif

            nec_bottom_up_compute_step(_graph, _levels, connections_array, current_level, vis, in_lvl, _use_vect_CSR_extension);

            #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
            t_end = omp_get_wtime();
            #endif
        }

        prev_frontier_size = current_frontier_size;
        current_frontier_size = vis;

        current_state = nec_change_state(prev_frontier_size, current_frontier_size, vertices_count, edges_count, current_state,
                                         vis, in_lvl, _use_vect_CSR_extension, current_level, graph_structure, _levels);
        current_level++;

        #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
        step_times.push_back(t_end - t_st);
        #endif
    }
    double t2 = omp_get_wtime();

    double compute_time = t2 - t1;
    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    compute_time = 0;
    for(int i = 0; i < step_times.size(); i++)
    {
        cout << "step " << i << " perf: " << edges_count/(step_times[i]*1e6) << " MTEPS, time: " << 1000.0 * step_times[i] << " ms, " << " % in state " << step_states[i] << endl;
        compute_time += step_times[i];
    }
    cout << "time diff: " << compute_time << " vs " << t2 - t1 << endl;
    PerformanceStats::print_algorithm_performance_stats("BFS (direction-optimizing)", compute_time, edges_count, current_level);
    #endif

    MemoryAPI::free_array(connections_array);

    double inner_perf = edges_count / (compute_time*1e6);
    return inner_perf;
}
#endif*/

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _T>
void BFS::nec_direction_optimizing(VectCSRGraph &_graph,
                                   VerticesArray<_T> &_levels,
                                   int _source_vertex)
{
    GraphAbstractionsNEC graph_API(_graph);
    FrontierNEC frontier(_graph);
    int vertices_count = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count();

    graph_API.change_traversal_direction(SCATTER, _levels, frontier);
    _source_vertex = _graph.reorder(_source_vertex, ORIGINAL, SCATTER);

    frontier.set_all_active();
    auto init_levels = [_levels, _source_vertex] (int src_id, int connections_count, int vector_index)
    {
        if(src_id == _source_vertex)
            _levels[_source_vertex] = FIRST_LEVEL_VERTEX;
        else
            _levels[src_id] = UNVISITED_VERTEX;
    };
    graph_API.compute(_graph, frontier, init_levels);

    frontier.clear();
    frontier.add_vertex(_source_vertex);

    int vis = 1, in_lvl = 0;
    int current_level = FIRST_LEVEL_VERTEX;
    StateOfBFS current_state = TOP_DOWN;
    bool _use_vect_CSR_extension = false;
    int current_frontier_size = 1, prev_frontier_size = 0;
    while(vis > 0)
    {
        vis = 0, in_lvl = 0;
        if(current_state == TOP_DOWN)
        {
            cout << "in TD state" << endl;

            #pragma omp parallel
            {
                NEC_REGISTER_INT(vis, 0);
                NEC_REGISTER_INT(in_lvl, 0);

                auto edge_op_with_stats = [_levels, current_level, &reg_vis, &reg_in_lvl](int src_id, int dst_id, int local_edge_pos,
                long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
                {
                    int dst_level = _levels[dst_id];
                    reg_in_lvl[vector_index]++;
                    if(dst_level == UNVISITED_VERTEX)
                    {
                        _levels[dst_id] = current_level + 1;
                        reg_vis[vector_index]++;
                    }
                };

                graph_API.scatter(_graph, frontier, edge_op_with_stats);

                int local_vis = register_sum_reduce(reg_vis);
                int local_in_lvl = register_sum_reduce(reg_in_lvl);

                #pragma omp atomic
                vis += local_vis;

                #pragma omp atomic
                in_lvl += local_in_lvl;
            }

            cout << "vis TD: " << vis << endl;
            cout << "in lvl TD: " << in_lvl << endl;
        }
        else if(current_state == BOTTOM_UP)
        {
            cout << "in BU state" << endl;

            auto is_unvisited = [_levels] (int src_id, int connections_count)->int
            {
                int result = NOT_IN_FRONTIER_FLAG;
                if((_levels[src_id] == UNVISITED_VERTEX) && (connections_count > 0))
                    result = IN_FRONTIER_FLAG;
                return result;
            };
            graph_API.generate_new_frontier(_graph, frontier, is_unvisited);

            frontier.print_stats();

            #pragma omp parallel
            {
                NEC_REGISTER_INT(vis, 0);
                NEC_REGISTER_INT(in_lvl, 0);

                int *levels_ptr = _levels.get_ptr();

                auto edge_op = [levels_ptr, current_level, &reg_vis, &reg_in_lvl](int src_id, int dst_id, int local_edge_pos,
                    long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
                {
                    //reg_in_lvl[vector_index]++;
                    if((levels_ptr[src_id] == UNVISITED_VERTEX) && (levels_ptr[dst_id] == current_level))
                    {
                        levels_ptr[src_id] = current_level + 1;
                        reg_vis[vector_index]++;
                    }
                };

                auto edge_collective_op = [levels_ptr, current_level, &reg_vis, &reg_in_lvl](int src_id, int dst_id, int local_edge_pos,
                    long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
                {
                    //reg_in_lvl[vector_index]++;
                    if((levels_ptr[src_id] == UNVISITED_VERTEX) && (levels_ptr[dst_id] == current_level))
                    {
                        levels_ptr[src_id] = current_level + 1;
                        reg_vis[vector_index]++;
                    }
                };

                graph_API.scatter(_graph, frontier, edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP, edge_collective_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);

                int local_vis = register_sum_reduce(reg_vis);
                int local_in_lvl = register_sum_reduce(reg_in_lvl);

                #pragma omp atomic
                vis += local_vis;

                #pragma omp atomic
                in_lvl += local_in_lvl;
            }
            cout << "vis BU: " << vis << endl;
            cout << "in lvl BU: " << in_lvl << endl;
        }

        prev_frontier_size = current_frontier_size;
        current_frontier_size = vis;

        current_state = nec_change_state(prev_frontier_size, current_frontier_size, vertices_count, edges_count, current_state,
                                         vis, in_lvl, _use_vect_CSR_extension, current_level, POWER_LAW_GRAPH, _levels.get_ptr());
        if(current_state == TOP_DOWN)
        {
            auto on_next_level = [&_levels, current_level] (int src_id, int connections_count)->int
            {
                int result = NOT_IN_FRONTIER_FLAG;
                if(_levels[src_id] == (current_level + 1))
                    result = IN_FRONTIER_FLAG;
                return result;
            };
            graph_API.generate_new_frontier(_graph, frontier, on_next_level);
        }
        current_level++;
    }

    cout << "iterations count: " << current_level << endl;
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _T>
void BFS::nec_top_down(VectCSRGraph &_graph,
                       VerticesArray<_T> &_levels,
                       int _source_vertex)
{
    GraphAbstractionsNEC graph_API(_graph);
    FrontierNEC frontier(_graph);

    graph_API.change_traversal_direction(SCATTER, _levels, frontier);

    _source_vertex = _graph.reorder(_source_vertex, ORIGINAL, SCATTER);

    #pragma omp parallel
    {};

    auto init_levels = [&_levels, _source_vertex] (int src_id, int connections_count, int vector_index)
    {
        if(src_id == _source_vertex)
            _levels[_source_vertex] = FIRST_LEVEL_VERTEX;
        else
            _levels[src_id] = UNVISITED_VERTEX;
    };
    frontier.set_all_active();
    graph_API.compute(_graph, frontier, init_levels);

    frontier.clear();
    frontier.add_vertex(_source_vertex);

    Timer tm;
    tm.start();

    int current_level = FIRST_LEVEL_VERTEX;
    while(frontier.size() > 0)
    {
        auto edge_op = [&_levels, &current_level](int src_id, int dst_id, int local_edge_pos,
                long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
        {
            int src_level = _levels[src_id];
            int dst_level = _levels[dst_id];
            if((src_level == current_level) && (dst_level == UNVISITED_VERTEX))
            {
                _levels[dst_id] = current_level + 1;
            }
        };

        graph_API.scatter(_graph, frontier, edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP,
                          edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);

        auto on_next_level = [&_levels, current_level] (int src_id, int connections_count)->int
        {
            int result = NOT_IN_FRONTIER_FLAG;
            if(_levels[src_id] == (current_level + 1))
                result = IN_FRONTIER_FLAG;
            return result;
        };

        graph_API.generate_new_frontier(_graph, frontier, on_next_level);

        current_level++;
    }

    tm.end();

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    PerformanceStats::print_algorithm_performance_stats("BFS (Top-down, NEC)", tm.get_time(), _graph.get_edges_count(), current_level);
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


