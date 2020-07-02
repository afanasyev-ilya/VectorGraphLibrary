#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _TVertexValue, typename _TEdgeWeight>
void BFS<_TVertexValue, _TEdgeWeight>::nec_top_down_compute_step(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                                 int *_levels,
                                                                 int _current_level,
                                                                 int &_vis,
                                                                 int &_in_lvl,
                                                                 bool _compute_stats)
{
    if(_compute_stats)
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
    }
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _TVertexValue, typename _TEdgeWeight>
void BFS<_TVertexValue, _TEdgeWeight>::nec_bottom_up_compute_step(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                                  int *_levels,
                                                                  int *_connections_array,
                                                                  int _current_level,
                                                                  int &_vis,
                                                                  int &_in_lvl,
                                                                  bool _use_vector_extension)
{
    int first_edge = 0;
    if(_use_vector_extension)
    {
        frontier.set_all_active();
        /*auto vertex_value_is_unset = [_levels, _connections_array] (int src_id)->int
        {
            int result = NOT_IN_FRONTIER_FLAG;
            if((_levels[src_id] == UNVISITED_VERTEX) && (_connections_array[src_id] > 0))
                result = IN_FRONTIER_FLAG;
            return result;
        };
        graph_API.generate_new_frontier(_graph, frontier, vertex_value_is_unset);*/

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
    }
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _TVertexValue, typename _TEdgeWeight>
void BFS<_TVertexValue, _TEdgeWeight>::nec_top_down(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                    int *_levels,
                                                    int _source_vertex)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);
    frontier.set_all_active();

    auto init_levels = [_levels, _source_vertex] (int src_id, int connections_count, int vector_index)
    {
        if(src_id == _source_vertex)
            _levels[_source_vertex] = FIRST_LEVEL_VERTEX;
        else
            _levels[src_id] = UNVISITED_VERTEX;
    };
    graph_API.compute(_graph, frontier, init_levels);

    frontier.add_vertex(_graph, _source_vertex);

    double t1 = omp_get_wtime();
    int current_level = FIRST_LEVEL_VERTEX;
    while(frontier.size() > 0)
    {
        auto edge_op = [_levels, current_level](int src_id, int dst_id, int local_edge_pos,
                long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
        {
            int src_level = _levels[src_id];
            int dst_level = _levels[dst_id];
            if((src_level == current_level) && (dst_level == UNVISITED_VERTEX))
            {
                _levels[dst_id] = current_level + 1;
            }
        };

        auto on_next_level = [_levels, current_level] (int src_id)->int
        {
            int result = NOT_IN_FRONTIER_FLAG;
            if(_levels[src_id] == (current_level + 1))
                result = IN_FRONTIER_FLAG;
            return result;
        };

        graph_API.advance(_graph, frontier, frontier, edge_op, on_next_level);

        current_level++;
    }
    double t2 = omp_get_wtime();

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    PerformanceStats::print_performance_stats("BFS (top-down)", t2 - t1, edges_count, current_level);
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define VERTICES_IN_VECTOR_EXTENSION 4

void init_vector_extension(const long long *_vertex_pointers,
                           const int *_adjacent_ids,
                           const int _vertices_count,
                           int *_vector_extension)
{
    for(int src_id = 0; src_id < _vertices_count; src_id++)
    {
        long long start = _vertex_pointers[src_id];
        long long end = _vertex_pointers[src_id + 1];
        int connections_count = end - start;
        for (int i = 0; i < min(connections_count, VERTICES_IN_VECTOR_EXTENSION); i++)
        {
            _vector_extension[_vertices_count * i + src_id] = _adjacent_ids[start + i];
        }
    }
}

#ifdef __USE_NEC_SX_AURORA__
template <typename _TVertexValue, typename _TEdgeWeight>
double BFS<_TVertexValue, _TEdgeWeight>::nec_direction_optimizing(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                                int *_levels,
                                                                int _source_vertex)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);
    GraphStructure graph_structure = check_graph_structure(_graph);
    frontier.set_all_active();

    int *vector_extension;
    MemoryAPI::allocate_array(&vector_extension, vertices_count * VERTICES_IN_VECTOR_EXTENSION);
    init_vector_extension(_graph.get_outgoing_ptrs(), _graph.get_outgoing_ids(), vertices_count, vector_extension);

    double t1 = omp_get_wtime();

    auto init_levels = [_levels, _source_vertex] (int src_id, int connections_count, int vector_index)
    {
        if(src_id == _source_vertex)
            _levels[_source_vertex] = FIRST_LEVEL_VERTEX;
        else
            _levels[src_id] = UNVISITED_VERTEX;
    };
    graph_API.compute(_graph, frontier, init_levels);

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
            #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
            t_st = omp_get_wtime();
            #endif

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

                graph_API.advance(_graph, frontier, edge_op_with_stats);

                int local_vis = register_sum_reduce(reg_vis);
                int local_in_lvl = register_sum_reduce(reg_in_lvl);

                #pragma omp atomic
                vis += local_vis;

                #pragma omp atomic
                in_lvl += local_in_lvl;
            }

            #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
            t_end = omp_get_wtime();
            #endif
        }
        else if(current_state == BOTTOM_UP)
        {
            #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
            t_st = omp_get_wtime();
            #endif

            double ve_t1 = omp_get_wtime();
            for(int i = 0; i < VERTICES_IN_VECTOR_EXTENSION; i++)
            {
                #pragma _NEC ivdep
                #pragma _NEC vovertake
                #pragma _NEC novob
                #pragma _NEC vector
                #pragma omp parallel for
                for(int src_id = 0; src_id < vertices_count; src_id++)
                {
                    if(_levels[src_id] == UNVISITED_VERTEX)
                    {
                        long long start = outgoing_ptrs[src_id];
                        long long end = outgoing_ptrs[src_id + 1];
                        int connections_count = end - start;
                        for(int i = 0; i < VERTICES_IN_VECTOR_EXTENSION; i++)
                        {
                            int dst_id = vector_extension[i * vertices_count + src_id];
                            if ((i < connections_count) && (_levels[dst_id] == current_level))
                            {
                                _levels[src_id] = current_level + 1;
                            }
                        }
                    }
                }
            }
            double ve_t2 = omp_get_wtime();

            auto is_unvisited = [_levels] (int src_id)->int
            {
                if(_levels[src_id] == UNVISITED_VERTEX)
                    return IN_FRONTIER_FLAG;
                else
                    return NOT_IN_FRONTIER_FLAG;
            };
            graph_API.generate_new_frontier(_graph, frontier, is_unvisited);

            #pragma omp parallel
            {
                NEC_REGISTER_INT(vis, 0);
                NEC_REGISTER_INT(in_lvl, 0);

                auto edge_op = [_levels, current_level, &reg_vis, &reg_in_lvl](int src_id, int dst_id, int local_edge_pos,
                        long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
                {
                    reg_in_lvl[vector_index]++;
                    if((_levels[src_id] == UNVISITED_VERTEX) && (_levels[dst_id] == current_level))
                    {
                        _levels[src_id] = current_level + 1;
                        reg_vis[vector_index]++;
                    }
                };

                auto edge_collective_op = [_levels, current_level, &reg_vis, &reg_in_lvl](int src_id, int dst_id, int local_edge_pos,
                        long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
                {
                    reg_in_lvl[vector_index]++;
                    if((_levels[src_id] == UNVISITED_VERTEX) && (_levels[dst_id] == current_level))
                    {
                        _levels[src_id] = current_level + 1;
                        reg_vis[vector_index]++;
                    }
                };

                graph_API.advance(_graph, frontier, edge_op);

                int local_vis = register_sum_reduce(reg_vis);
                int local_in_lvl = register_sum_reduce(reg_in_lvl);

                #pragma omp atomic
                vis += local_vis;

                #pragma omp atomic
                in_lvl += local_in_lvl;
            }

            #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
            t_end = omp_get_wtime();
            #endif
        }

        prev_frontier_size = current_frontier_size;
        current_frontier_size = vis;

        current_state = nec_change_state(prev_frontier_size, current_frontier_size, vertices_count, edges_count, current_state,
                                         vis, in_lvl, _use_vect_CSR_extension, current_level, graph_structure, _levels);

        if(current_state == TOP_DOWN)
        {
            auto on_next_level = [_levels, current_level] (int src_id)->int
            {
                if(_levels[src_id] == (current_level + 1))
                    return IN_FRONTIER_FLAG;
                else
                    return NOT_IN_FRONTIER_FLAG;
            };
            graph_API.generate_new_frontier(_graph, frontier, on_next_level);
        }
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
    PerformanceStats::print_performance_stats("BFS (direction-optimizing)", compute_time, edges_count, current_level);
    #endif

    double inner_perf = edges_count / (compute_time*1e6);
    return inner_perf;

    MemoryAPI::free_array(vector_extension);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

