#pragma once

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

    auto calculate_non_zero_count = []__VGL_COMPUTE_ARGS__->int
    {
        int result = 0;
        if(connections_count > 0)
        {
            result = 1;
        }
        return result;
    };
    int non_zero_count = graph_API.reduce<int>(_graph, frontier, calculate_non_zero_count, REDUCE_SUM);

    auto set_reminder = [_levels, non_zero_count]__VGL_COMPUTE_ARGS__
    {
        if(src_id >= non_zero_count)
            _levels[src_id] = UNVISITED_VERTEX;
    };
    graph_API.compute(_graph, frontier, set_reminder);

    frontier.change_size(non_zero_count);
    frontier.set_all_active();

    auto init_connections = [connections_array] __VGL_COMPUTE_ARGS__
    {
        connections_array[src_id] = connections_count;
    };
    graph_API.compute(_graph, frontier, init_connections);

    double t1 = omp_get_wtime();

    auto init_levels = [_levels, _source_vertex] __VGL_COMPUTE_ARGS__
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
    performance_stats.print_algorithm_performance_stats("BFS (direction-optimizing)", compute_time, edges_count, current_level);
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
                                   int _source_vertex,
                                   BFS_GraphVE &_vector_extension,
                                   DirectionType _direction,
                                   int first_switch,
                                   int second_switch)
{
    GraphAbstractionsNEC graph_API(_graph, SCATTER);
    FrontierNEC frontier(_graph, SCATTER);

    int *segment_needs_processing;
    MemoryAPI::allocate_array(&segment_needs_processing, _vector_extension.ve_vertices_count / VECTOR_LENGTH);

    int vertices_count = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count();

    _source_vertex = 4000; //_graph.reorder(_source_vertex, ORIGINAL, SCATTER);

    vector<StateOfBFS> step_states;
    vector<double> step_times;

    Timer tm;
    tm.start();
    auto init_levels = [_levels, _source_vertex] __VGL_COMPUTE_ARGS__
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

    double bu_ve_time = 0, bu_time = 0, td_time = 0, gnf_time = 0;
    int BU_count = 0;
    int vis = 1, in_lvl = 0;
    int current_level = FIRST_LEVEL_VERTEX;
    StateOfBFS current_state = TOP_DOWN;
    bool _use_vect_CSR_extension = false;
    int current_frontier_size = 1, prev_frontier_size = 0;
    while(vis > 0)
    {
        Timer tm_step;
        tm_step.start();
        step_states.push_back(current_state);

        vis = 0, in_lvl = 0;

        if(current_state == TOP_DOWN)
        {
            Timer tm_td;
            tm_td.start();

            //cout << " -------------- IN TOP DOWN STATE ----------------- " << endl;
            int *levels_ptr = _levels.get_ptr();

            Timer tm_gnf;
            tm_gnf.start();
            if(current_level > FIRST_LEVEL_VERTEX)
            {
                auto on_this_level = [levels_ptr, current_level] (int src_id, int connections_count)->int
                {
                    int result = NOT_IN_FRONTIER_FLAG;
                    if(levels_ptr[src_id] == (current_level))
                        result = IN_FRONTIER_FLAG;
                    return result;
                };
                graph_API.generate_new_frontier(_graph, frontier, on_this_level);
            }
            tm_gnf.end();
            gnf_time +=tm_gnf.get_time_in_ms();

            auto edge_op = [_levels, &current_level](int src_id, int dst_id, int local_edge_pos,
                long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
            {
                int dst_level = _levels[dst_id];
                if(dst_level == UNVISITED_VERTEX)
                {
                    _levels[dst_id] = current_level + 1;
                }
            };

            //frontier.print_stats();
            graph_API.scatter(_graph, frontier, edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP,
                              edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);
            //cout << "inner: " << (t2 - t1)*1000 << " ms" << endl;

            vis = frontier.size();
            in_lvl = frontier.get_neighbours_count();

            //cout << "vis TD: " << vis << " vs " << frontier.size() << endl;
            //cout << "in lvl TD: " << in_lvl << " vs " << frontier.get_neighbours_count() << endl;

            tm_td.end();
            td_time += tm_td.get_time_in_ms();
        }
        else if(current_state == BOTTOM_UP)
        {
            //cout << " -------------- IN BOTTOM UP STATE ----------------- " << endl;

            BU_count++;
            int *levels_ptr = _levels.get_ptr();

            if(BU_count <= 2)
            {
                Timer tm_ve;
                tm_ve.start();
                int ve_vertices_count = _vector_extension.ve_vertices_count;
                int *ve_dst_ids = _vector_extension.ve_dst_ids;

                int ve_starting_vertex = 0;
                int ve_vector_segments_count = ve_vertices_count / VECTOR_LENGTH;

                #pragma omp parallel
                {
                    NEC_REGISTER_INT(vis, 0);
                    NEC_REGISTER_INT(in_lvl, 0);

                    #pragma omp for schedule(static, 8)
                    for(int cur_vector_segment = 0; cur_vector_segment < ve_vector_segments_count; cur_vector_segment++)
                    {
                        int segment_first_vertex = cur_vector_segment * VECTOR_LENGTH;
                        long long segment_edges_start = cur_vector_segment * VECTOR_LENGTH * BFS_VE_SIZE;
                        int segment_connections_count = BFS_VE_SIZE;

                        for(int edge_pos = 0; edge_pos < segment_connections_count; edge_pos++)
                        {
                            #pragma _NEC cncall
                            #pragma _NEC ivdep
                            #pragma _NEC vovertake
                            #pragma _NEC novob
                            #pragma _NEC vector
                            #pragma _NEC gather_reorder
                            for (int i = 0; i < VECTOR_LENGTH; i++)
                            {
                                const int src_id = segment_first_vertex + i;
                                int src_level = levels_ptr[src_id];

                                if(src_level == UNVISITED_VERTEX)
                                {
                                    const long long internal_edge_pos = segment_edges_start + edge_pos * VECTOR_LENGTH + i;
                                    const int dst_id = ve_dst_ids[internal_edge_pos];

                                    int dst_level = 0;
                                    if(dst_id != -1)
                                    {
                                        dst_level = levels_ptr[dst_id];
                                        reg_in_lvl[i]++;
                                    }

                                    if((dst_id != -1) && (dst_level == current_level))
                                    {
                                        levels_ptr[src_id] = current_level + 1;
                                        reg_vis[i]++;
                                    }
                                }
                            }
                        }
                    }

                    int local_vis = register_sum_reduce(reg_vis);
                    int local_in_lvl = register_sum_reduce(reg_in_lvl);

                    #pragma omp atomic
                    vis += local_vis;

                    #pragma omp atomic
                    in_lvl += local_in_lvl;
                }
                tm_ve.end();
                performance_stats.update_non_api_time(tm_ve);
                //tm_ve.print_time_and_bandwidth_stats("BFS VE v7", ve_vertices_count * BFS_VE_SIZE, sizeof(int)*3.0);
                //cout << tm_ve.get_time_in_ms() << " (ms) VE time" << endl;
                bu_ve_time += tm_ve.get_time_in_ms();
            }

            auto is_unvisited = [_levels] (int src_id, int connections_count)->int
            {
                int result = NOT_IN_FRONTIER_FLAG;
                if((_levels[src_id] == UNVISITED_VERTEX) && (connections_count >= BFS_VE_SIZE))
                    result = IN_FRONTIER_FLAG;
                return result;
            };
            Timer tm_my;
            tm_my.start();
            graph_API.generate_new_frontier(_graph, frontier, is_unvisited);
            tm_my.end();
            //tm_my.print_time_stats("gen time BU front");

            int front_size = frontier.size();
            int *front_ids = frontier.get_ids();
            long long    *vertex_pointers = _graph.get_outgoing_graph_ptr()->get_vertex_pointers();
            int          *adjacent_ids = _graph.get_outgoing_graph_ptr()->get_adjacent_ids();
            int vc_border = frontier.get_vector_engine_part_size() + frontier.get_vector_core_part_size();

            tm_my.start();

            #pragma omp parallel
            {
                int reg_in_lvl[VECTOR_LENGTH];
                int reg_vis[VECTOR_LENGTH];
                int reg_connections[VECTOR_LENGTH];
                int reg_starts[VECTOR_LENGTH];

                #pragma _NEC vreg(reg_connections)
                #pragma _NEC vreg(reg_starts)
                #pragma _NEC vreg(reg_in_lvl)
                #pragma _NEC vreg(reg_vis)

                #pragma _NEC vector
                for(int i = 0; i < VECTOR_LENGTH; i++)
                {
                    reg_in_lvl[i] = 0;
                    reg_vis[i] = 0;
                }

                #pragma omp for schedule(static, 8)
                for(int front_pos = 0; front_pos < vc_border; front_pos++)
                {
                    int src_id = front_ids[front_pos];
                    int connections_count = vertex_pointers[src_id + 1] - vertex_pointers[src_id];
                    long long start = vertex_pointers[src_id];

                    #pragma _NEC cncall
                    #pragma _NEC ivdep
                    #pragma _NEC vovertake
                    #pragma _NEC novob
                    #pragma _NEC vector
                    #pragma _NEC gather_reorder
                    for(int edge_pos = BFS_VE_SIZE; edge_pos < connections_count; edge_pos++)
                    {
                        int dst_id = adjacent_ids[start + edge_pos];
                        if(levels_ptr[dst_id] == current_level)
                        {
                            levels_ptr[src_id] = current_level + 1;
                        }
                    }
                }

                #pragma omp for schedule(static)
                for(int front_pos = vc_border; front_pos < front_size; front_pos += VECTOR_LENGTH)
                {
                    int max_connections = 0;
                    #pragma _NEC vector
                    #pragma _NEC ivdep
                    for(int i = 0; i < VECTOR_LENGTH; i++)
                    {
                        if((front_pos + i) < front_size)
                        {
                            int src_id = front_ids[front_pos + i];
                            int connections_count = vertex_pointers[src_id + 1] - vertex_pointers[src_id];
                            if(max_connections < connections_count)
                                max_connections = connections_count;

                            reg_connections[i] = connections_count;
                            reg_starts[i] = vertex_pointers[src_id];
                        }
                    }

                    for(int edge_pos = BFS_VE_SIZE; edge_pos < max_connections; edge_pos++)
                    {
                        #pragma _NEC cncall
                        #pragma _NEC ivdep
                        #pragma _NEC vovertake
                        #pragma _NEC novob
                        #pragma _NEC vector
                        #pragma _NEC gather_reorder
                        for(int i = 0; i < VECTOR_LENGTH; i++)
                        {
                            if((front_pos + i) < front_size)
                            {
                                int src_id = front_ids[front_pos + i];
                                int real_connections_count = reg_connections[i];
                                long long start = reg_starts[i];
                                reg_in_lvl[i]++;

                                if(real_connections_count < max_connections)
                                {
                                    int dst_id = adjacent_ids[start + edge_pos];
                                    if(levels_ptr[dst_id] == current_level)
                                    {
                                        levels_ptr[src_id] = current_level + 1;
                                        reg_vis[i]++;
                                    }
                                }
                            }
                        }
                    }
                }

                int local_vis = register_sum_reduce(reg_vis);
                int local_in_lvl = register_sum_reduce(reg_in_lvl);

                #pragma omp atomic
                vis += local_vis;

                #pragma omp atomic
                in_lvl += local_in_lvl;
            }

            /*#pragma omp parallel
            {
                NEC_REGISTER_INT(vis, 0);
                NEC_REGISTER_INT(in_lvl, 0);
                NEC_REGISTER_INT(levels, 0);

                auto edge_op = [_levels, current_level, &reg_vis, &reg_in_lvl, &reg_levels] (int src_id, int dst_id, int local_edge_pos,
                    long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
                {
                    reg_in_lvl[vector_index]++;
                    if(_levels[dst_id] == current_level)
                    {
                        //levels_ptr[src_id] = current_level + 1;
                        reg_levels[vector_index] = current_level + 1;
                        reg_vis[vector_index]++;
                    }
                };

                auto postprocess = [_levels, current_level, &reg_levels] (int src_id, int connections_count, int vector_index, DelayedWriteNEC &delayed_write)
                {
                    int new_level = _levels[src_id];
                    #pragma _NEC vector
                    #pragma _NEC ivdep
                    for(int i = 0; i < VECTOR_LENGTH; i++)
                        if(reg_levels[i] == (current_level + 1))
                            new_level = reg_levels[i];

                    _levels[src_id] = new_level;
                };

                auto edge_collective_op = [_levels, current_level, &reg_vis, &reg_in_lvl](int src_id, int dst_id, int local_edge_pos,
                    long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
                {
                    reg_in_lvl[vector_index]++;
                    if(_levels[dst_id] == current_level)
                    {
                        _levels[src_id] = current_level + 1;
                        reg_vis[vector_index]++;
                    }
                };

                if(_direction == UNDIRECTED_GRAPH)
                {
                    graph_API.scatter(_graph, frontier, edge_op, EMPTY_VERTEX_OP, postprocess, edge_collective_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);
                }
                else
                {
                    graph_API.gather(_graph, frontier, edge_op, EMPTY_VERTEX_OP, postprocess, edge_collective_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);
                }

                int local_vis = register_sum_reduce(reg_vis);
                int local_in_lvl = register_sum_reduce(reg_in_lvl);

                #pragma omp atomic
                vis += local_vis;

                #pragma omp atomic
                in_lvl += local_in_lvl;
            }*/

            tm_my.end();
            performance_stats.update_non_api_time(tm_my);
            long long work = frontier.get_vector_engine_part_neighbours_count() + frontier.get_vector_core_part_neighbours_count() + frontier.get_collective_part_neighbours_count() - BFS_VE_SIZE*front_size;
            //tm_my.print_time_and_bandwidth_stats("new BU BFS", work, sizeof(int)*3.0);
            bu_time += tm_my.get_time_in_ms();
            //cout << tm_my.get_time_in_ms() << " (ms) VGL sparse time" << endl;
        }

        prev_frontier_size = current_frontier_size;
        current_frontier_size = vis;

        /*StateOfBFS new_state = nec_change_state(prev_frontier_size, current_frontier_size, vertices_count, edges_count, current_state,
                                         vis, in_lvl, _use_vect_CSR_extension, current_level, POWER_LAW_GRAPH, _levels.get_ptr());

        if(BU_count == 2)
        {
            new_state = TOP_DOWN;
        }

        if((_direction == DIRECTED_GRAPH) && (new_state != current_state))
        {
            if(new_state == BOTTOM_UP)
                graph_API.change_traversal_direction(GATHER, _levels, frontier);
            if(new_state == TOP_DOWN)
                graph_API.change_traversal_direction(SCATTER, _levels, frontier);
        }*/
        //StateOfBFS new_state;
        if(current_level == first_switch)
            current_state = BOTTOM_UP;
        if(current_level == second_switch)
            current_state = TOP_DOWN;

        //current_state = new_state;

        //current_state = man_states[current_level];

        current_level++;
        tm_step.end();
        step_times.push_back(tm_step.get_time());
    }
    tm.end();

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    cout << "iterations count: " << current_level << endl;
    for(int i = 0; i < step_times.size(); i++)
    {
        cout << "step " << i << " perf: " << edges_count/(step_times[i]*1e6) << " MTEPS, time: " << 1000.0 * step_times[i] << " ms, " << " % in state " << step_states[i] << endl;
    }
    cout << "BU ve time: " << bu_ve_time << " ms" << endl;
    cout << "BU time: " << bu_time << " ms" << endl;
    cout << "TD time: " << td_time << " ms" << endl;
    cout << "GNF time: " << gnf_time << " ms" << endl;
    performance_stats.print_algorithm_performance_stats("BFS (Direction-optimizing, NEC)", tm.get_time(), _graph.get_edges_count(), current_level);
    #endif
    cout << _graph.get_edges_count() / (1e6*tm.get_time()) << " MTEPS" << endl;
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
template <typename _T>
void BFS::nec_top_down(VectCSRGraph &_graph,
                       VerticesArray<_T> &_levels,
                       int _source_vertex)
{
    VGL_GRAPH_ABSTRACTIONS graph_API(_graph);
    VGL_FRONTIER frontier(_graph);

    graph_API.change_traversal_direction(SCATTER, _levels, frontier);

    _source_vertex = _graph.reorder(_source_vertex, ORIGINAL, SCATTER);

    #pragma omp parallel
    {};

    auto init_levels = [&_levels, _source_vertex] __VGL_COMPUTE_ARGS__
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

    performance_stats.save_algorithm_performance_stats(tm.get_time(), _graph.get_edges_count());
    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    performance_stats.print_algorithm_performance_stats("BFS (Top-down, NEC/multicore)");
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


