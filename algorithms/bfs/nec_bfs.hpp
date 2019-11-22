//
//  nec_bfs.hpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 21/10/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef nec_bfs_h
#define nec_bfs_h

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
inline void BFS<_TVertexValue, _TEdgeWeight>::nec_top_down_step(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                                long long *_outgoing_ptrs,
                                                                int *_outgoing_ids,
                                                                int _vertices_count,
                                                                int _active_count,
                                                                int *_levels,
                                                                int *_cached_levels,
                                                                int _cur_level,
                                                                int &_vis,
                                                                int &_in_lvl,
                                                                int _threads_count)
{
    #pragma _NEC retain(_levels)
    #pragma _NEC retain(_cached_levels)
    
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
                
                int *private_levels = _graph.template get_private_data_pointer<int>(_cached_levels);
                
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
                        int dst_level = _graph.template load_vertex_data_cached<int>(dst_id, _levels, private_levels);
                        
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
                        int dst_level = _graph.template load_vertex_data_cached<int>(dst_id, _levels, private_levels);
                        
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
            
            int *private_levels = _graph.template get_private_data_pointer<int>(_cached_levels);
            
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
                        int dst_level = _graph.template load_vertex_data_cached<int>(dst_id, _levels, private_levels);
                        
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
                        int dst_level = _graph.template load_vertex_data_cached<int>(dst_id, _levels, private_levels);
                        
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
                    int dst_level = _graph.template load_vertex_data_cached<int>(dst_id, _levels, private_levels);
                    
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
                    #ifdef __USE_NEC_SX_AURORA__
                    #pragma _NEC ivdep
                    #pragma _NEC vovertake
                    #pragma _NEC novob
                    #pragma _NEC vector
                    #endif
                    for(int i = 0; i < VECTOR_LENGTH; i++)
                    {
                        int src_id = active_reg[i];
                        int dst_id = _outgoing_ids[start_pos_reg[i] + edge_pos];
                        int dst_level = 0;
                        
                        if(((vec_start + i) < _active_count) && (edge_pos < connections_reg[i]))
                        {
                            in_lvl_reg[i]++;
                            dst_level = _graph.template load_vertex_data_cached<int>(dst_id, _levels, private_levels);
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

template <typename _TVertexValue, typename _TEdgeWeight>
inline void BFS<_TVertexValue, _TEdgeWeight>::nec_bottom_up_step(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                                 long long *_outgoing_ptrs,
                                                                 int *_outgoing_ids,
                                                                 int _vertices_count,
                                                                 int _active_count,
                                                                 int *_levels,
                                                                 int *_cached_levels,
                                                                 int _cur_level,
                                                                 int &_vis,
                                                                 int &_in_lvl,
                                                                 int _threads_count,
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
        #pragma omp parallel shared(updated_count, vector_iterations_count)
        {
            int *private_levels = _graph.template get_private_data_pointer<int>(_cached_levels);
            
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
                    #pragma _NEC ivdep
                    #pragma _NEC vovertake
                    #pragma _NEC novob
                    #pragma _NEC vector
                    for(int i = 0; i < VECTOR_LENGTH; i++)
                    {
                        int src_id = vec_start + i;
                        if(_levels[src_id] == UNVISITED_VERTEX)
                        {
                            int connections = _outgoing_ptrs[src_id + 1] - _outgoing_ptrs[src_id];
                            int dst_id = vectorised_outgoing_ids[src_id + step * _non_zero_vertices_count];
                            int dst_level = _graph.template load_vertex_data_cached<int>(dst_id, _levels, private_levels);
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
        
    int active_vertices_left = sparse_copy_if(_levels, active_ids, active_vertices_buffer, _non_zero_vertices_count,
                                              BOTTOM_UP_REMINDER_VERTEX, _threads_count);
    
    t2 = omp_get_wtime();
    _t_second = t2 - t1;
    
    t1 = omp_get_wtime();
    #pragma omp parallel shared(vis, in_lvl)
    {
        long long local_in_lvl = 0;
        int local_vis = 0;
            
        int connections[VECTOR_LENGTH];
        int active_reg[VECTOR_LENGTH];
        long long start_pos[VECTOR_LENGTH];
        int vis_reg[VECTOR_LENGTH];
        long long in_lvl_reg[VECTOR_LENGTH];
        int src_levels_reg[VECTOR_LENGTH];
            
        #pragma _NEC vreg(start_pos)
        #pragma _NEC vreg(connections)
        #pragma _NEC vreg(active_reg)
        #pragma _NEC vreg(vis_reg)
        #pragma _NEC vreg(in_lvl_reg)
        #pragma _NEC vreg(src_levels_reg)
            
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            connections[i] = 0;
            active_reg[i] = 0;
            start_pos[i] = 0;
            in_lvl_reg[i] = 0;
            vis_reg[i] = 0;
            src_levels_reg[i] = 0;
        }
            
        int *private_levels = _graph.template get_private_data_pointer<int>(_cached_levels);
            
        #pragma _NEC novector
        #pragma omp for schedule(static, 1)
        for(int vec_start = 0; vec_start < active_vertices_left; vec_start += VECTOR_LENGTH)
        {
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                int src_id = active_ids[vec_start + i];
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
                if(max_connections < connections[i])
                    max_connections = connections[i];
            }
                
            for(int edge_pos = vector_iterations_count; edge_pos < max_connections; edge_pos++)
            {
                #pragma _NEC ivdep
                #pragma _NEC vovertake
                #pragma _NEC novob
                #pragma _NEC vector
                for(int i = 0; i < VECTOR_LENGTH; i++)
                {
                    int src_id = active_reg[i];
                    int dst_id = _outgoing_ids[start_pos[i] + edge_pos];
                    int dst_level = 0;
                        
                    if(((vec_start + i) < active_vertices_left) && (edge_pos < connections[i]))
                    {
                        dst_level = _graph.template load_vertex_data_cached<int>(dst_id, _levels, private_levels);
                        in_lvl_reg[i]++;
                    }

                    if(((vec_start + i) < active_vertices_left) && (edge_pos < connections[i]) && (dst_level == _cur_level))
                    {
                        src_levels_reg[i] = _cur_level + 1;
                        vis_reg[i]++;
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

template <typename _TVertexValue, typename _TEdgeWeight>
inline int BFS<_TVertexValue, _TEdgeWeight>::nec_remove_zero_nodes(long long *_outgoing_ptrs, int _vertices_count, int *_levels)
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

template <typename _TVertexValue, typename _TEdgeWeight>
inline int BFS<_TVertexValue, _TEdgeWeight>::nec_mark_zero_nodes(int _vertices_count, int *_levels)
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

template <typename _TVertexValue, typename _TEdgeWeight>
void BFS<_TVertexValue, _TEdgeWeight>::nec_direction_optimising_BFS(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                                    int *_levels,
                                                                    int _source_vertex)
{
    double t1, t2, t3, t4;
    int vertices_count    = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count   ();
    long long    *outgoing_ptrs    = _graph.get_outgoing_ptrs   ();
    int          *outgoing_ids     = _graph.get_outgoing_ids    ();
     
    GraphStructure graph_structure = check_graph_structure(_graph);
    
    int threads_count = omp_get_max_threads();
    _graph.set_threads_count(threads_count);
    
    int cur_level = 1, active_count = 1;
    _graph.template vertex_array_set_to_constant<int>(_levels, UNVISITED_VERTEX);
    _graph.template vertex_array_set_element<int>(_levels, _source_vertex,  cur_level);
    _graph.template vertex_array_set_to_constant<int>(active_ids, 0);
    _graph.template vertex_array_set_element<int>(active_ids, 0, _source_vertex);
    
    int *cached_levels = _graph.template allocate_private_caches<int>(threads_count);
    
    StateOfBFS current_state = TOP_DOWN;
    double total_time = 0;
    
    t1 = omp_get_wtime();
    int non_zero_vertices_count = nec_remove_zero_nodes(outgoing_ptrs, vertices_count, _levels);
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
        #pragma omp parallel num_threads(threads_count)
        {
            int *private_levels = _graph.template get_private_data_pointer<int>(cached_levels);
            _graph.template place_data_into_cache<int>(_levels, private_levels);
        }
        
        int vis = 0, in_lvl = 0;
        int current_active_count = active_count;
        
        if(current_state == TOP_DOWN)
        {
            nec_top_down_step(_graph, outgoing_ptrs, outgoing_ids, vertices_count, active_count, _levels,
                              cached_levels, cur_level, vis, in_lvl, threads_count);
        }
        else if(current_state == BOTTOM_UP)
        {
            //use_vect_CSR_extension = true;
            nec_bottom_up_step(_graph, outgoing_ptrs, outgoing_ids, vertices_count, active_count, _levels, cached_levels,
                               cur_level, vis, in_lvl, threads_count, use_vect_CSR_extension,
                               non_zero_vertices_count, t_first, t_second, t_third);
        }
        
        t4 = omp_get_wtime();
        total_time += t4 - t3;
        double kernel_time = t4 - t3;
        
        total_kernel_time += kernel_time;
        
        t3 = omp_get_wtime();
        int next_active_count = get_elements_count(_levels, non_zero_vertices_count, cur_level + 1);
        int frontier_size = next_active_count;
        
        if(frontier_size == 0)
            break;
        
        StateOfBFS next_state = change_state(current_active_count, next_active_count, vertices_count, edges_count, current_state,
                                             vis, in_lvl, use_vect_CSR_extension, cur_level, graph_structure);
        
        if(next_state == TOP_DOWN)
        {
            active_count = sparse_copy_if(_levels, active_ids, active_vertices_buffer, non_zero_vertices_count, cur_level + 1,
                                          threads_count);
        }
        else if(next_state == BOTTOM_UP)
        {
            active_count = get_elements_count(_levels, non_zero_vertices_count, UNVISITED_VERTEX);
        }
        t4 = omp_get_wtime();
        double reminder_time = t4 - t3;
        total_time += reminder_time;
        
        #ifdef PRINT_DETAILED_STATS
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
    
    t1 = omp_get_wtime();
    nec_mark_zero_nodes(vertices_count, _levels);
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
    cout << "TOTAL BFS Perf: " << ((double)edges_count)/(total_time*1e6) << " MTEPS" << endl << endl << endl;
    #endif
    
    _graph.template free_data<int>(cached_levels);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* nec_bfs_h */
