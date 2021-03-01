#pragma once

#define BOTTOM_UP_THRESHOLD 5
#define PRINT_DETAILED_STATS

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define ALPHA 15
#define BETA 18

#define POWER_LAW_EDGES_THRESHOLD 30

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

GraphStructure check_graph_structure(VectCSRGraph &_graph)
{
    int vertices_count    = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count   ();
    long long    *outgoing_ptrs = _graph.get_outgoing_graph_ptr()->get_vertex_pointers();
    int          *outgoing_ids = _graph.get_outgoing_graph_ptr()->get_adjacent_ids();

    int portion_of_first_vertices = 0.01 * vertices_count + 1;
    long long number_of_edges_in_first_portion = outgoing_ptrs[portion_of_first_vertices];

    if((100.0 * number_of_edges_in_first_portion) / edges_count > POWER_LAW_EDGES_THRESHOLD)
        return POWER_LAW_GRAPH;
    else
        return UNIFORM_GRAPH;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int nec_remove_zero_nodes(long long *_outgoing_ptrs, int _vertices_count, int *_levels)
{
    int zero_nodes_count = 0;
    #pragma _NEC vector
    #pragma omp parallel for schedule(static) reduction(+: zero_nodes_count)
    for(int src_id = 0; src_id < _vertices_count; src_id++)
    {
        int connections = _outgoing_ptrs[src_id + 1] - _outgoing_ptrs[src_id];
        if(connections == 0)
        {
            _levels[src_id] = ISOLATED_VERTEX;
            zero_nodes_count++;
        }
    }
    return _vertices_count - zero_nodes_count;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void nec_mark_zero_nodes(int _vertices_count, int *_levels)
{
    #pragma _NEC vector
    #pragma omp parallel for schedule(static)
    for(int src_id = 0; src_id < _vertices_count; src_id++)
    {
        if(_levels[src_id] == ISOLATED_VERTEX)
        {
            _levels[src_id] = UNVISITED_VERTEX;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void nec_top_down_step(long long *_outgoing_ptrs,
                                                                int *_outgoing_ids,
                                                                int _vertices_count,
                                                                int _active_count,
                                                                int *_levels,
                                                                int _cur_level,
                                                                int &_vis,
                                                                int &_in_lvl,
                                                                int _threads_count,
                                                                int *active_ids)
{
    #pragma _NEC retain(_levels)

    int vis = 0, in_lvl = 0;
    if(_cur_level == 1) // process first level vertex
    {
        int src_id = active_ids[0];
        int src_level = _levels[src_id];
        int connections = _outgoing_ptrs[src_id + 1] - _outgoing_ptrs[src_id];
        int start_pos = _outgoing_ptrs[src_id];
        in_lvl += connections;

        if(connections < VECTOR_LENGTH) // process first level vertex with low degree
        {
            #pragma _NEC ivdep
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            for(int edge_pos = 0; edge_pos < connections; edge_pos++)
            {
                int dst_id = _outgoing_ids[start_pos + edge_pos];
                if((src_level == _cur_level) && (_levels[dst_id] == UNVISITED_VERTEX))
                {
                    _levels[dst_id] = _cur_level + 1;
                    vis++;
                }
            }
        }
        else // process first vertex with large degree
        {
            #pragma omp parallel
            {
                int local_vis_reg[VECTOR_LENGTH];

                #pragma _NEC vreg(local_vis_reg)

                for(int i = 0; i < VECTOR_LENGTH; i++)
                    local_vis_reg[i] = 0;

                //int *private_levels = _graph.template get_private_data_pointer<int>(_cached_levels); TODO

                #pragma _NEC novector
                #pragma omp for schedule(static)
                for(int edge_pos = 0; edge_pos < connections - VECTOR_LENGTH; edge_pos += VECTOR_LENGTH)
                {
                    #pragma _NEC ivdep
                    #pragma _NEC vovertake
                    #pragma _NEC novob
                    #pragma _NEC vector
                    for(int i = 0; i < VECTOR_LENGTH; i++)
                    {
                        int dst_id = _outgoing_ids[start_pos + edge_pos + i];
                        int dst_level = _levels[dst_id];
                                //_graph.template load_vertex_data_cached<int>(dst_id, _levels, private_levels);

                        if(dst_level == UNVISITED_VERTEX)
                        {
                            _levels[dst_id] = _cur_level + 1;
                            local_vis_reg[i]++;
                        }
                    }
                }
                #pragma omp single
                {
                    #pragma _NEC ivdep
                    #pragma _NEC vovertake
                    #pragma _NEC novob
                    #pragma _NEC vector
                    for(int edge_pos = connections - VECTOR_LENGTH; edge_pos < connections; edge_pos++)
                    {
                        int i = edge_pos - (connections - VECTOR_LENGTH);
                        int dst_id = _outgoing_ids[start_pos + edge_pos];
                        int dst_level = _levels[dst_id];
                        // = _graph.template load_vertex_data_cached<int>(dst_id, _levels, private_levels);

                        if(dst_level == UNVISITED_VERTEX)
                        {
                            _levels[dst_id] = _cur_level + 1;
                            local_vis_reg[i]++;
                        }
                    }
                }

                int local_vis = 0;
                for(int i = 0; i < VECTOR_LENGTH; i++)
                    local_vis += local_vis_reg[i];

                #pragma omp barrier

                #pragma omp critical
                {
                    vis += local_vis;
                }
            }
        }
    }
    else // process full layer of vertices
    {
        int border_large, border_medium = 0;
        #pragma omp parallel shared(vis, in_lvl, border_large, border_medium)
        {
            int local_vis = 0;
            long long local_in_lvl = 0;

            int connections_reg[VECTOR_LENGTH];
            int active_reg[VECTOR_LENGTH];
            long long start_pos_reg[VECTOR_LENGTH];
            int vis_reg[VECTOR_LENGTH];
            int in_lvl_reg[VECTOR_LENGTH];

            #pragma _NEC vreg(start_pos_reg)
            #pragma _NEC vreg(connections_reg)
            #pragma _NEC vreg(active_reg)
            #pragma _NEC vreg(vis_reg)
            #pragma _NEC vreg(in_lvl_reg)

            // int *private_levels = _graph.template get_private_data_pointer<int>(_cached_levels); TODO

            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                connections_reg[i] = 0;
                active_reg[i] = 0;
                start_pos_reg[i] = 0;
                in_lvl_reg[i] = 0;
                vis_reg[i] = 0;
            }

            // estimate borders
            #pragma omp for schedule(static)
            for(int i = 0; i < (_active_count - 1); i++)
            {
                int id1 = active_ids[i];
                int id2 = active_ids[i + 1];
                int connections1 = _outgoing_ptrs[id1 + 1] - _outgoing_ptrs[id1];
                int connections2 = _outgoing_ptrs[id2 + 1] - _outgoing_ptrs[id2];
                if((connections1 > 32*VECTOR_LENGTH) && (connections2 <= 32*VECTOR_LENGTH))
                {
                    border_large = i;
                }

                if((connections1 > VECTOR_LENGTH) && (connections2 <= VECTOR_LENGTH))
                {
                    border_medium = i;
                }
            }

            #pragma omp barrier

            // process group of "large" vertices
            #pragma _NEC novector
            for(int idx = 0; idx < border_large; idx++)
            {
                #pragma omp barrier

                int src_id = active_ids[idx];
                int src_level = _levels[src_id];
                int connections = _outgoing_ptrs[src_id + 1] - _outgoing_ptrs[src_id];
                int start_pos = _outgoing_ptrs[src_id];

                #pragma omp master
                {
                    local_in_lvl += connections;
                }

                #pragma _NEC novector
                #pragma omp for schedule(static)
                for(int edge_pos = 0; edge_pos < connections - VECTOR_LENGTH; edge_pos += VECTOR_LENGTH)
                {
                    #pragma _NEC ivdep
                    #pragma _NEC vovertake
                    #pragma _NEC novob
                    #pragma _NEC vector
                    for(int i = 0; i < VECTOR_LENGTH; i++)
                    {
                        int dst_id = _outgoing_ids[start_pos + edge_pos + i];
                        int dst_level = _levels[dst_id];
                        //int dst_level = _graph.template load_vertex_data_cached<int>(dst_id, _levels, private_levels);

                        if(dst_level == UNVISITED_VERTEX)
                        {
                            _levels[dst_id] = _cur_level + 1;
                            vis_reg[i]++;
                        }
                    }
                }
                #pragma omp single
                {
                    #pragma _NEC ivdep
                    #pragma _NEC vovertake
                    #pragma _NEC novob
                    #pragma _NEC vector
                    for(int edge_pos = connections - VECTOR_LENGTH; edge_pos < connections; edge_pos++)
                    {
                        int i = edge_pos - (connections - VECTOR_LENGTH);
                        int dst_id = _outgoing_ids[start_pos + edge_pos];
                        int dst_level = _levels[dst_id];
                        //int dst_level = _graph.template load_vertex_data_cached<int>(dst_id, _levels, private_levels);

                        if(dst_level == UNVISITED_VERTEX)
                        {
                            _levels[dst_id] = _cur_level + 1;
                            vis_reg[i]++;
                        }
                    }
                }
            }

            // traverse group of "medium" vertices
            #pragma _NEC novector
            #pragma omp for schedule(static)
            for(int idx = border_large; idx < border_medium; idx++)
            {
                int src_id = active_ids[idx];
                int connections = _outgoing_ptrs[src_id + 1] - _outgoing_ptrs[src_id];
                long long start_pos = _outgoing_ptrs[src_id];
                local_in_lvl += connections;

                #pragma _NEC ivdep
                #pragma _NEC vovertake
                #pragma _NEC novob
                #pragma _NEC vector
                for(int edge_pos = 0; edge_pos < connections; edge_pos++)
                {
                    int i = edge_pos % VECTOR_LENGTH;
                    int dst_id = _outgoing_ids[start_pos + edge_pos];
                    int dst_level = _levels[dst_id];

                    if(dst_level == UNVISITED_VERTEX)
                    {
                        _levels[dst_id] = _cur_level + 1;
                        vis_reg[i]++;
                    }
                }
            }

            #pragma omp for schedule(static)
            for(int vec_start = border_medium; vec_start < _active_count; vec_start += VECTOR_LENGTH)
            {
                for(int i = 0; i < VECTOR_LENGTH; i++)
                {
                    int src_id = active_ids[vec_start + i];
                    active_reg[i] = src_id;

                    if((vec_start + i) < _active_count)
                    {
                        connections_reg[i] = _outgoing_ptrs[src_id + 1] - _outgoing_ptrs[src_id];
                        start_pos_reg[i] = _outgoing_ptrs[src_id];
                    }
                    else
                    {
                        connections_reg[i] = 0;
                        start_pos_reg[i] = 0;
                    }
                }

                int max_connections = 0;
                for(int i = 0; i < VECTOR_LENGTH; i++)
                {
                    if(max_connections < connections_reg[i])
                        max_connections = connections_reg[i];
                }

                for(int edge_pos = 0; edge_pos < max_connections; edge_pos++)
                {
                    #pragma _NEC ivdep
                    #pragma _NEC vovertake
                    #pragma _NEC novob
                    #pragma _NEC vector
                    for(int i = 0; i < VECTOR_LENGTH; i++)
                    {
                        int src_id = active_reg[i];
                        int dst_id = _outgoing_ids[start_pos_reg[i] + edge_pos];
                        int dst_level = 0;

                        if(((vec_start + i) < _active_count) && (edge_pos < connections_reg[i]))
                        {
                            in_lvl_reg[i]++;
                            dst_level = _levels[dst_id];
                        }

                        if(((vec_start + i) < _active_count) && (edge_pos < connections_reg[i]) && (dst_level == UNVISITED_VERTEX))
                        {
                            _levels[dst_id] = _cur_level + 1;
                            vis_reg[i]++;
                        }
                    }
                }
            }

            #pragma _NEC vector
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                local_vis += vis_reg[i];
                local_in_lvl += in_lvl_reg[i];
            }

            #pragma omp barrier

            #pragma omp critical
            {
                vis += local_vis;
                in_lvl += local_in_lvl;
            }

        }
    }
    _vis = vis;
    _in_lvl = in_lvl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline int sparse_copy_if(int *_in_data,
                          int *_out_data,
                          int *_tmp_buffer,
                          int _size,
                          int _desired_value,
                          int _threads_count = MAX_SX_AURORA_THREADS)
{
    int elements_per_thread = _size/_threads_count;
    int elements_per_vector = elements_per_thread/VECTOR_LENGTH;
    int shifts_array[MAX_SX_AURORA_THREADS];

    int elements_count = 0;
    #pragma omp parallel num_threads(_threads_count) shared(elements_count)
    {
        int tid = omp_get_thread_num();
        int start_pointers_reg[VECTOR_LENGTH];
        int current_pointers_reg[VECTOR_LENGTH];
        int last_pointers_reg[VECTOR_LENGTH];

        #pragma _NEC vreg(start_pointers_reg)
        #pragma _NEC vreg(current_pointers_reg)
        #pragma _NEC vreg(last_pointers_reg)

        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            start_pointers_reg[i] = tid * elements_per_thread + i * elements_per_vector;
            current_pointers_reg[i] = tid * elements_per_thread + i * elements_per_vector;
            last_pointers_reg[i] = tid * elements_per_thread + i * elements_per_vector;
        }

        #pragma omp for schedule(static)
        for(int vec_start = 0; vec_start < _size; vec_start += VECTOR_LENGTH)
        {
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                int src_id = vec_start + i;
                if(_in_data[src_id] == _desired_value)
                {
                    _tmp_buffer[current_pointers_reg[i]] = src_id;
                    current_pointers_reg[i]++;
                }
            }
        }

        int max_difference = 0;
        int save_values_per_thread = 0;
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            int difference = current_pointers_reg[i] - start_pointers_reg[i];
            save_values_per_thread += difference;
            if(difference > max_difference)
                max_difference = difference;
        }

        shifts_array[tid] = save_values_per_thread;
        #pragma omp barrier

        #pragma omp master
        {
            int cur_shift = 0;
            for(int i = 1; i < _threads_count; i++)
            {
                shifts_array[i] += shifts_array[i - 1];
            }

            elements_count = shifts_array[_threads_count - 1];

            for(int i = (_threads_count - 1); i >= 1; i--)
            {
                shifts_array[i] = shifts_array[i - 1];
            }
            shifts_array[0] = 0;
        }

        #pragma omp barrier

        int tid_shift = shifts_array[tid];
        int *private_ptr = &(_out_data[tid_shift]);

        int local_pos = 0;
        #pragma _NEC novector
        for(int pos = 0; pos < max_difference; pos++)
        {
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                int size = current_pointers_reg[i] - start_pointers_reg[i];
                if(pos < size)
                {
                    private_ptr[local_pos] = _tmp_buffer[last_pointers_reg[i]];
                    last_pointers_reg[i]++;
                    local_pos++;
                }
            }
        }
    }

    return elements_count;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void man_opt_nec_bfs(VectCSRGraph &_graph, VerticesArray<int> &_levels, int _source_vertex)
{
    double t1, t2, t3, t4;
    int vertices_count    = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count   ();
    long long    *outgoing_ptrs = _graph.get_outgoing_graph_ptr()->get_vertex_pointers();
    int          *outgoing_ids = _graph.get_outgoing_graph_ptr()->get_adjacent_ids();
    int *levels_ptr = _levels.get_ptr();
    int *active_ids, *active_vertices_buffer;
    MemoryAPI::allocate_array(&active_ids, vertices_count);
    MemoryAPI::allocate_array(&active_vertices_buffer, vertices_count);

    GraphStructure graph_structure = check_graph_structure(_graph);

    cout << "structure: " << graph_structure << endl;

    int threads_count = omp_get_max_threads();

    int cur_level = 1, active_count = 1;
    for(int i = 0; i < vertices_count; i++)
    {
        levels_ptr[i] = UNVISITED_VERTEX;
        active_ids[i] = 0;
    }
    levels_ptr[_source_vertex] = cur_level;
    active_ids[0] = _source_vertex;

    StateOfBFS current_state = TOP_DOWN;
    double total_time = 0;

    t1 = omp_get_wtime();
    int non_zero_vertices_count = nec_remove_zero_nodes(outgoing_ptrs, vertices_count, levels_ptr);
    t2 = omp_get_wtime();
    total_time += t2 - t1;

    vector<double> each_kernel_time;
    vector<double> each_remider_time;
    vector<double> each_first_time, each_second_time, each_third_time;
    vector<StateOfBFS> each_step_type;

    double total_kernel_time = 0, total_reminder_time = 0;

    bool use_vect_CSR_extension = false;
    while(active_count > 0)
    {
        double t_first, t_second, t_third;
        t3 = omp_get_wtime();
        /*#pragma omp parallel num_threads(threads_count)
        {
            int *private_levels = _graph.template get_private_data_pointer<int>(cached_levels);
            _graph.template place_data_into_cache<int>(_levels, private_levels);
        }*/

        int vis = 0, in_lvl = 0;
        int current_active_count = active_count;

        if(current_state == TOP_DOWN)
        {
            nec_top_down_step(outgoing_ptrs, outgoing_ids, vertices_count, active_count, levels_ptr,
                              cur_level, vis, in_lvl, threads_count, active_ids);
            cout << "done!" << endl;
        }
        else if(current_state == BOTTOM_UP)
        {
            break;
            /*nec_bottom_up_step(_graph, outgoing_ptrs, outgoing_ids, vertices_count, active_count, _levels, cached_levels,
                               cur_level, vis, in_lvl, threads_count, use_vect_CSR_extension,
                               non_zero_vertices_count, t_first, t_second, t_third);*/
        }

        t4 = omp_get_wtime();
        total_time += t4 - t3;
        double kernel_time = t4 - t3;

        total_kernel_time += kernel_time;

        t3 = omp_get_wtime();
        int next_active_count = get_elements_count(levels_ptr, non_zero_vertices_count, cur_level + 1);
        int frontier_size = next_active_count;

        if(frontier_size == 0)
            break;

        StateOfBFS next_state = nec_change_state(current_active_count, next_active_count, vertices_count, edges_count, current_state,
                                             vis, in_lvl, use_vect_CSR_extension, cur_level, graph_structure, levels_ptr);

        if(next_state == TOP_DOWN)
        {
            active_count = sparse_copy_if(levels_ptr, active_ids, active_vertices_buffer, non_zero_vertices_count, cur_level + 1,
                                          threads_count);
        }
        else if(next_state == BOTTOM_UP)
        {
            active_count = get_elements_count(levels_ptr, non_zero_vertices_count, UNVISITED_VERTEX);
        }
        t4 = omp_get_wtime();
        double reminder_time = t4 - t3;
        total_time += reminder_time;

        total_reminder_time += reminder_time;
        each_kernel_time.push_back(kernel_time);
        each_remider_time.push_back(reminder_time);
        each_first_time.push_back(t_first);
        each_second_time.push_back(t_second);
        each_third_time.push_back(t_third);
        each_step_type.push_back(current_state);

        current_state = next_state;
        cur_level++;
        break;
    }

    t1 = omp_get_wtime();
    nec_mark_zero_nodes(vertices_count, levels_ptr);
    t2 = omp_get_wtime();
    total_time += t2 - t1;

    #ifdef PRINT_DETAILED_STATS
    for(int i = 0; i < each_kernel_time.size(); i++)
    {
        string state = "top_down";
        if(each_step_type[i] == BOTTOM_UP)
            state = "bottom_up";
        cout << "level " << i + 1 << " in " << state << " state" << endl;
        if(each_step_type[i] == BOTTOM_UP)
        {
            cout << "first: " << 100.0 * each_first_time[i]/total_time << " % of total time (" << 1000.0*each_first_time[i] << "ms)"  << endl;
            cout << "second: " << 100.0 * each_second_time[i]/total_time << " % of total time (" << 1000.0*each_second_time[i] << "ms)"  << endl;
            cout << "third: " << 100.0 * each_third_time[i]/total_time << " % of total time (" << 1000.0*each_third_time[i] << "ms)"  << endl;
        }
        cout << "KERNEL: " << 100.0 * each_kernel_time[i]/total_time << " % of total time (" << 1000.0*each_kernel_time[i] << "ms)"  << endl;
        cout << "REMINDER: " << 100.0 * each_remider_time[i]/total_time << " % of total time (" << 1000.0*each_remider_time[i] << "ms)" << endl << endl;
    }
    cout << "kernel input: " << 100.0*total_kernel_time/total_time << " %" << endl;
    cout << "reminder input: " << 100.0*total_reminder_time/total_time << " %" << endl << endl;
    cout << total_kernel_time << " vs " << total_reminder_time << endl;
    //cout << "TOTAL BFS Perf: " << ((double)edges_count)/(total_time*1e6) << " MTEPS" << endl << endl << endl;
    #endif
    cout << "TOTAL BFS Perf: " << ((double)edges_count)/(total_time*1e6) << " MTEPS" << endl << endl << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
