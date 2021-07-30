#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

GraphStructure check_graph_structure(VectCSRGraph &_graph)
{
    int vertices_count    = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count   ();
    long long    *outgoing_ptrs = _graph.get_outgoing_data()->get_vertex_pointers();
    int          *outgoing_ids = _graph.get_outgoing_data()->get_adjacent_ids();

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

inline int fast_sparse_copy_if(const int *_in_data,
                               int *_out_data,
                               int *_tmp_buffer,
                               const int _buffer_size,
                               const int _start,
                               const int _end,
                               const int _desired_value,
                               const int _threads_count = MAX_SX_AURORA_THREADS)
{
    int size = _end - _start;
    int elements_per_thread = (_buffer_size - 1)/_threads_count + 1;
    int elements_per_vector = (elements_per_thread - 1)/VECTOR_LENGTH + 1;
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

        #pragma _NEC vector
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            start_pointers_reg[i] = tid * elements_per_thread + i * elements_per_vector;
            current_pointers_reg[i] = tid * elements_per_thread + i * elements_per_vector;
            last_pointers_reg[i] = tid * elements_per_thread + i * elements_per_vector;
        }

        #pragma omp for schedule(static)
        for(int vec_start = _start; vec_start < _end; vec_start += VECTOR_LENGTH)
        {
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                int src_id = vec_start + i;
                if((src_id < _end) && (_in_data[src_id] == _desired_value))
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
                int loc_size = current_pointers_reg[i] - start_pointers_reg[i];

                if(pos < loc_size)
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


inline int new_sorted_copy_if(VectCSRGraph &_graph,
                              const int *_in_data,
                              int *_out_data,
                              int *_tmp_buffer,
                              const int _buffer_size,
                              const int _start,
                              const int _end,
                              const int _desired_value,
                              int &_large_border,
                              int &_medium_border)
{
    // get borders
    const int ve_threshold = _graph.get_outgoing_data()->get_vector_engine_threshold_vertex();
    int vc_threshold = _graph.get_outgoing_data()->get_vector_core_threshold_vertex();

    int first = fast_sparse_copy_if(_in_data, _out_data, _tmp_buffer, _buffer_size,
                                    _start, ve_threshold, _desired_value);

    _large_border = first;

    int second = fast_sparse_copy_if(_in_data, &_out_data[first], _tmp_buffer, _buffer_size,
                                     ve_threshold, vc_threshold, _desired_value);

    _medium_border = first + second;

    int third = fast_sparse_copy_if(_in_data, &_out_data[first + second], _tmp_buffer, _buffer_size,
                                    vc_threshold, _end, _desired_value);

    return first + second + third;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void hardwired_nec_top_down_step(long long *_outgoing_ptrs,
                           int *_outgoing_ids,
                           int _vertices_count,
                           int _active_count,
                           VerticesArray<_T> &_levels,
                           int _cur_level,
                           int &_vis,
                           int &_in_lvl,
                           int *active_ids,
                           int _large_border,
                           int _medium_border)
{
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

                int *private_levels = _levels.get_private_data_pointer();

                #pragma _NEC novector
                #pragma omp for schedule(static)
                for(int edge_pos = 0; edge_pos < connections; edge_pos += VECTOR_LENGTH)
                {
                    #pragma _NEC ivdep
                    #pragma _NEC vovertake
                    #pragma _NEC novob
                    #pragma _NEC vector
                    for(int i = 0; i < VECTOR_LENGTH; i++)
                    {
                        if((edge_pos + i) < connections)
                        {
                            int dst_id = _outgoing_ids[start_pos + edge_pos + i];
                            int dst_level = _levels.cached_load(dst_id, private_levels);

                            if(dst_level == UNVISITED_VERTEX)
                            {
                                _levels[dst_id] = _cur_level + 1;
                                local_vis_reg[i]++;
                            }
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

            int *private_levels = _levels.get_private_data_pointer();

            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                connections_reg[i] = 0;
                active_reg[i] = 0;
                start_pos_reg[i] = 0;
                in_lvl_reg[i] = 0;
                vis_reg[i] = 0;
            }

            border_large = _large_border;
            border_medium = _medium_border;

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
                for(int edge_pos = 0; edge_pos < connections; edge_pos += VECTOR_LENGTH)
                {
                    #pragma _NEC cncall
                    #pragma _NEC ivdep
                    #pragma _NEC vovertake
                    #pragma _NEC novob
                    #pragma _NEC vector
                    #pragma _NEC sparse
                    #pragma _NEC gather_reorder
                    for(int i = 0; i < VECTOR_LENGTH; i++)
                    {
                        if((edge_pos + i) < connections)
                        {
                            int dst_id = _outgoing_ids[start_pos + edge_pos + i];
                            int dst_level = _levels.cached_load(dst_id, private_levels);

                            if(dst_level == UNVISITED_VERTEX)
                            {
                                _levels[dst_id] = _cur_level + 1;
                                vis_reg[i]++;
                            }
                        }
                    }
                }
            }

            // traverse group of "medium" vertices
            #pragma _NEC novector
            #pragma omp for schedule(static, 8)
            for(int idx = border_large; idx < border_medium; idx++)
            {
                int src_id = active_ids[idx];
                int connections = _outgoing_ptrs[src_id + 1] - _outgoing_ptrs[src_id];
                long long start_pos = _outgoing_ptrs[src_id];
                local_in_lvl += connections;

                #pragma _NEC novector
                for(int edge_pos = 0; edge_pos < connections; edge_pos += VECTOR_LENGTH)
                {
                    #pragma _NEC cncall
                    #pragma _NEC ivdep
                    #pragma _NEC vovertake
                    #pragma _NEC novob
                    #pragma _NEC vector
                    #pragma _NEC sparse
                    #pragma _NEC gather_reorder
                    for(int i = 0; i < VECTOR_LENGTH; i++)
                    {
                        if ((edge_pos + i) < connections)
                        {
                            int dst_id = _outgoing_ids[start_pos + edge_pos + i];
                            int dst_level = _levels.cached_load(dst_id, private_levels);

                            if(dst_level == UNVISITED_VERTEX)
                            {
                                _levels[dst_id] = _cur_level + 1;
                                vis_reg[i]++;
                            }
                        }
                    }
                }
            }

            #pragma omp for schedule(static, 8)
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
                    #pragma _NEC cncall
                    #pragma _NEC ivdep
                    #pragma _NEC vovertake
                    #pragma _NEC novob
                    #pragma _NEC vector
                    #pragma _NEC sparse
                    #pragma _NEC gather_reorder
                    for(int i = 0; i < VECTOR_LENGTH; i++)
                    {
                        int src_id = active_reg[i];
                        int dst_id = _outgoing_ids[start_pos_reg[i] + edge_pos];
                        int dst_level = _levels.cached_load(dst_id, private_levels);

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
    //tm_old.end();
    //cout << "OLD time: " << tm_old.get_time_in_ms() << " ms" << endl;

    /*Timer tm_vgl;
    tm_vgl.start();
    #pragma omp parallel
    {
        int *private_levels = _levels.get_private_data_pointer();

        auto edge_op = [&_levels, _cur_level, private_levels](int src_id, int dst_id, int local_edge_pos,
                                             long long int global_edge_pos, int vector_index)
        {
            int dst_level = _levels.cached_load(dst_id, private_levels);
            if(dst_level == UNVISITED_VERTEX)
            {
                _levels[dst_id] = _cur_level + 1;
            }
        };
        graph_API.scatter(_graph, frontier, edge_op);
    }

    _vis = frontier.size();
    _in_lvl = frontier.get_neighbours_count();

    tm_vgl.end();
    //cout << "VGL BU time: " << tm_vgl.get_time_in_ms() << " ms" << endl;*/
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void hardwired_nec_bottom_up_step(VectCSRGraph &_graph,
                        long long *_outgoing_ptrs,
                        int *_outgoing_ids,
                        int *_vectorised_outgoing_ids,
                        int _vertices_count,
                        int *_active_ids,
                        int *_active_vertices_buffer,
                        int _active_count,
                        VerticesArray<_T> &_levels,
                        int _cur_level,
                        int &_vis,
                        int &_in_lvl,
                        bool _use_vect_CSR_extension,
                        int _non_zero_vertices_count,
                        double &_t_first, double &_t_second, double &_t_third)
{
    int vis = 0;
    long long in_lvl = 0;

    double t1, t2;

    int vector_iterations_count = 0;
    t1 = omp_get_wtime();
    if(_use_vect_CSR_extension)
    {
        int updated_count = 0;
        #pragma omp parallel shared(updated_count, vector_iterations_count, vis)
        {
            _T *private_levels = _levels.get_private_data_pointer();

            #pragma _NEC novector
            #pragma _NEC unroll(BOTTOM_UP_THRESHOLD)
            for(int step = 0; step < BOTTOM_UP_THRESHOLD; step++)
            {
                #pragma omp barrier

                updated_count = 0;

                int updated_reg[VECTOR_LENGTH];
                #pragma _NEC vreg(updated_reg)
                for(int i = 0; i < VECTOR_LENGTH; i++)
                    updated_reg[i] = 0;

                #pragma _NEC novector
                #pragma omp for schedule(static)
                for(int vec_start = 0; vec_start < _non_zero_vertices_count; vec_start += VECTOR_LENGTH)
                {
                    #pragma _NEC cncall
                    #pragma _NEC ivdep
                    #pragma _NEC vovertake
                    #pragma _NEC novob
                    #pragma _NEC vector
                    #pragma _NEC sparse
                    #pragma _NEC gather_reorder
                    for(int i = 0; i < VECTOR_LENGTH; i++)
                    {
                        int src_id = vec_start + i;
                        if(_levels[src_id] == UNVISITED_VERTEX)
                        {
                            int connections = _outgoing_ptrs[src_id + 1] - _outgoing_ptrs[src_id];
                            int dst_id = _vectorised_outgoing_ids[src_id + step * _non_zero_vertices_count];

                            int dst_level = _levels.cached_load(dst_id, private_levels);

                            if((dst_level == _cur_level) && (connections > step))
                            {
                                _levels[src_id] = _cur_level + 1;
                                updated_reg[i]++;
                            }
                        }
                    }
                }

                int local_updated = 0;
                for(int i = 0; i < VECTOR_LENGTH; i++)
                    local_updated += updated_reg[i];

                #pragma omp critical
                {
                    updated_count += local_updated;
                    vector_iterations_count = step + 1;
                }

                #pragma omp barrier

                #pragma omp single
                {
                    vis += updated_count;
                    in_lvl += _non_zero_vertices_count;
                }

                if ((100.0 * updated_count / _vertices_count) < 1)
                {
                    break;
                }
            }
        }
    }
    t2 = omp_get_wtime();
    _t_first = t2 - t1;

    t1 = omp_get_wtime();
    #pragma omp parallel for
    for(int i = 0; i < _vertices_count; i++)
    {
        if(_levels[i] == UNVISITED_VERTEX)
        {
            int connections = _outgoing_ptrs[i + 1] - _outgoing_ptrs[i];
            if(connections >= vector_iterations_count)
            {
                _levels[i] = BOTTOM_UP_REMINDER_VERTEX;
            }
        }
    }

    int active_vertices_left = 0;

    int large_border = 0, medium_border = 0;
    if(_cur_level != 2)
    {
        active_vertices_left = fast_sparse_copy_if(_levels.get_ptr(), _active_ids, _active_vertices_buffer,
                            _non_zero_vertices_count, 0, _non_zero_vertices_count, BOTTOM_UP_REMINDER_VERTEX);
    }
    else
    {
        active_vertices_left = new_sorted_copy_if(_graph, _levels.get_ptr(), _active_ids, _active_vertices_buffer,
                                                   _non_zero_vertices_count, 0, _non_zero_vertices_count, BOTTOM_UP_REMINDER_VERTEX, large_border, medium_border);
    }


    t2 = omp_get_wtime();
    _t_second = t2 - t1;

    t1 = omp_get_wtime();
    int border_large = 0;
    #pragma omp parallel shared(vis, in_lvl, border_large)
    {
        _T *private_levels = _levels.get_private_data_pointer();

        long long local_in_lvl = 0;
        int local_vis = 0;

        int connections[VECTOR_LENGTH];
        int active_reg[VECTOR_LENGTH];
        long long start_pos[VECTOR_LENGTH];
        int vis_reg[VECTOR_LENGTH];
        long long in_lvl_reg[VECTOR_LENGTH];
        int src_levels_reg[VECTOR_LENGTH];
        int processing_reg[VECTOR_LENGTH];

        #pragma _NEC vreg(start_pos)
        #pragma _NEC vreg(connections)
        #pragma _NEC vreg(active_reg)
        #pragma _NEC vreg(vis_reg)
        #pragma _NEC vreg(in_lvl_reg)
        #pragma _NEC vreg(src_levels_reg)
        #pragma _NEC vreg(processing_reg)

        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            connections[i] = 0;
            active_reg[i] = 0;
            start_pos[i] = 0;
            in_lvl_reg[i] = 0;
            vis_reg[i] = 0;
            src_levels_reg[i] = 0;
            processing_reg[i] = 0;
        }

        #pragma _NEC novector
        for(int idx = 0; idx < border_large; idx++)
        {
            int src_id = _active_ids[idx];
            int src_level = _levels[src_id];
            int connections = _outgoing_ptrs[src_id + 1] - _outgoing_ptrs[src_id];
            int start_pos = _outgoing_ptrs[src_id];

            #pragma _NEC novector
            #pragma omp for schedule(static)
            for(int edge_pos = 0; edge_pos < connections; edge_pos += VECTOR_LENGTH)
            {
                #pragma _NEC cncall
                #pragma _NEC ivdep
                #pragma _NEC vovertake
                #pragma _NEC novob
                #pragma _NEC vector
                #pragma _NEC sparse
                #pragma _NEC gather_reorder
                for(int i = 0; i < VECTOR_LENGTH; i++)
                {
                    if((edge_pos + i) < connections)
                    {
                        int dst_id = _outgoing_ids[start_pos + edge_pos + i];
                        int dst_level = _levels.cached_load(dst_id, private_levels);
                        in_lvl_reg[i]++;

                        if(dst_level == _cur_level)
                        {
                            _levels[src_id] = _cur_level + 1;
                            vis_reg[i]++;
                        }
                    }
                }
            }
        }

        #pragma _NEC novector
        #pragma omp for schedule(static, 8)
        for(int idx = large_border; idx < medium_border; idx++) // process "medium" vertices
        {
            int src_id = _active_ids[idx];
            int connections = _outgoing_ptrs[src_id + 1] - _outgoing_ptrs[src_id];
            long long start_pos = _outgoing_ptrs[src_id];

            #pragma _NEC novector
            for(int edge_pos = 0; edge_pos < connections; edge_pos += VECTOR_LENGTH)
            {
                #pragma _NEC cncall
                #pragma _NEC ivdep
                #pragma _NEC vovertake
                #pragma _NEC novob
                #pragma _NEC vector
                #pragma _NEC sparse
                #pragma _NEC gather_reorder
                for(int i = 0; i < VECTOR_LENGTH; i++)
                {
                    if ((edge_pos + i) < connections)
                    {
                        int dst_id = _outgoing_ids[start_pos + edge_pos + i];
                        int dst_level = _levels.cached_load(dst_id, private_levels);
                        in_lvl_reg[i]++;

                        if(dst_level == _cur_level)
                        {
                            _levels[src_id] = _cur_level + 1;
                            vis_reg[i]++;
                        }
                    }
                }
            }
        }

        #pragma _NEC novector
        #pragma omp for schedule(static, 8)
        for(int vec_start = medium_border; vec_start < active_vertices_left; vec_start += VECTOR_LENGTH) // process "small" vertices
        {
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                int src_id = _active_ids[vec_start + i];
                active_reg[i] = src_id;

                if((vec_start + i) < active_vertices_left)
                {
                    connections[i] = _outgoing_ptrs[src_id + 1] - _outgoing_ptrs[src_id];
                    start_pos[i] = _outgoing_ptrs[src_id];
                    src_levels_reg[i] = _levels[src_id];
                }
                else
                {
                    connections[i] = 0;
                    start_pos[i] = 0;
                    src_levels_reg[i] = 0;
                }
            }

            int max_connections = 0;
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                processing_reg[i] = 0;
                if(max_connections < connections[i])
                    max_connections = connections[i];
            }

            for(int edge_pos = vector_iterations_count; edge_pos < max_connections; edge_pos++)
            {
                #pragma _NEC cncall
                #pragma _NEC ivdep
                #pragma _NEC vovertake
                #pragma _NEC novob
                #pragma _NEC vector
                #pragma _NEC sparse
                #pragma _NEC gather_reorder
                for(int i = 0; i < VECTOR_LENGTH; i++)
                {
                    int src_id = active_reg[i];
                    int dst_id = _outgoing_ids[start_pos[i] + edge_pos];
                    int dst_level = 0;

                    if(((vec_start + i) < active_vertices_left) && (edge_pos < connections[i]))
                    {
                        dst_level = _levels.cached_load(dst_id, private_levels);
                        in_lvl_reg[i]++;
                    }

                    if(((vec_start + i) < active_vertices_left) && (edge_pos < connections[i]) && (dst_level == _cur_level))
                    {
                        src_levels_reg[i] = _cur_level + 1;
                    }
                }
            }

            #pragma _NEC ivdep
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                if((vec_start + i) < active_vertices_left)
                {
                    int src_id = active_reg[i];
                    if(src_levels_reg[i] != BOTTOM_UP_REMINDER_VERTEX)
                    {
                        _levels[src_id] = src_levels_reg[i];
                        vis_reg[i]++;
                    }
                    else
                    {
                        _levels[src_id] = UNVISITED_VERTEX;
                    }
                }
            }
        }

        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            local_in_lvl += in_lvl_reg[i];
            local_vis += vis_reg[i];
        }

        #pragma omp barrier

        #pragma omp critical
        {
            vis += local_vis;
            in_lvl += local_in_lvl;
        }
    }
    t2 = omp_get_wtime();
    _t_third = t2 - t1;

    _vis = vis;
    _in_lvl = in_lvl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void BFS::hardwired_do_bfs(VectCSRGraph &_graph,
                           VerticesArray<_T> &_levels,
                           int _source_vertex,
                           BFS_GraphVE &_vector_extension,
                           int *_buffer1,
                           int *_buffer2)
{
    _source_vertex = _graph.reorder(_source_vertex, ORIGINAL, SCATTER);

    double t1, t2, t3, t4;
    int vertices_count    = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count   ();
    long long    *outgoing_ptrs = _graph.get_outgoing_data()->get_vertex_pointers();
    int          *outgoing_ids = _graph.get_outgoing_data()->get_adjacent_ids();
    int *active_ids = _buffer1;
    int *active_vertices_buffer = _buffer2;

    GraphStructure graph_structure = check_graph_structure(_graph);

    int cur_level = FIRST_LEVEL_VERTEX, active_count = 1;
    #pragma omp parallel for
    for(int i = 0; i < vertices_count; i++)
    {
        _levels[i] = UNVISITED_VERTEX;
        active_ids[i] = 0;
    }
    _levels[_source_vertex] = cur_level;
    active_ids[0] = _source_vertex;

    #pragma omp parallel
    {};

    StateOfBFS current_state = TOP_DOWN;

    double total_time = 0;

    Timer tm;
    tm.start();
    int non_zero_vertices_count = nec_remove_zero_nodes(outgoing_ptrs, vertices_count, _levels.get_ptr());
    tm.end();
    total_time += tm.get_time();
    performance_stats.update_non_api_time(tm);

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    vector<double> each_kernel_time;
    vector<double> each_remider_time;
    vector<double> each_first_time, each_second_time, each_third_time;
    vector<StateOfBFS> each_step_type;
    #endif

    double total_kernel_time = 0, total_reminder_time = 0;
    int large_border = 0, medium_border = 0;

    int number_of_bu_steps = 0;
    bool use_vect_CSR_extension = false;
    while(active_count > 0)
    {
        double t_first, t_second, t_third;
        tm.start();
        _levels.prefetch_data_into_cache();

        int vis = 0, in_lvl = 0;
        int current_active_count = active_count;

        if(current_state == TOP_DOWN)
        {
            hardwired_nec_top_down_step(outgoing_ptrs, outgoing_ids, vertices_count, active_count, _levels,
                                  cur_level, vis, in_lvl, active_ids, large_border, medium_border);
        }
        else if(current_state == BOTTOM_UP)
        {
            hardwired_nec_bottom_up_step(_graph, outgoing_ptrs, outgoing_ids, _vector_extension.ve_dst_ids, vertices_count, active_ids,
                               active_vertices_buffer, active_count, _levels,
                               cur_level, vis, in_lvl, use_vect_CSR_extension,
                               non_zero_vertices_count, t_first, t_second, t_third);
            number_of_bu_steps++;
        }
        tm.end();
        total_time += tm.get_time();
        double kernel_time = tm.get_time();
        performance_stats.update_non_api_time(tm);

        tm.start();
        int next_active_count = get_elements_count(_levels.get_ptr(), non_zero_vertices_count, cur_level + 1);
        int frontier_size = next_active_count;

        if(frontier_size == 0)
            break;

        StateOfBFS next_state = nec_change_state(current_active_count, next_active_count, vertices_count, edges_count, current_state,
                                             vis, in_lvl, use_vect_CSR_extension, cur_level, graph_structure, _levels.get_ptr(), number_of_bu_steps);

        if(next_state == TOP_DOWN)
        {
            active_count = new_sorted_copy_if(_graph, _levels.get_ptr(), active_ids, active_vertices_buffer,
                                              non_zero_vertices_count, 0, non_zero_vertices_count, cur_level + 1, large_border, medium_border);
        }
        else if(next_state == BOTTOM_UP)
        {
            active_count = get_elements_count(_levels.get_ptr(), non_zero_vertices_count, UNVISITED_VERTEX);
        }
        tm.end();
        double reminder_time = tm.get_time();
        total_time += tm.get_time();
        performance_stats.update_non_api_time(tm);

        #ifdef __PRINT_API_PERFORMANCE_STATS__
        total_reminder_time += reminder_time;
        each_kernel_time.push_back(kernel_time);
        each_remider_time.push_back(reminder_time);
        each_first_time.push_back(t_first);
        each_second_time.push_back(t_second);
        each_third_time.push_back(t_third);
        each_step_type.push_back(current_state);
        #endif

        current_state = next_state;
        cur_level++;
    }

    tm.start();
    nec_mark_zero_nodes(vertices_count, _levels.get_ptr());
    tm.end();
    total_time += tm.get_time();
    performance_stats.update_non_api_time(tm);

    #ifdef __PRINT_API_PERFORMANCE_STATS__
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
    #endif

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    performance_stats.print_algorithm_performance_stats("BFS (Direction-optimizing, NEC/multicore)", total_time, _graph.get_edges_count());
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
