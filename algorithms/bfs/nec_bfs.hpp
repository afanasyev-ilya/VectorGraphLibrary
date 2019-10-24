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
inline int BFS<_TVertexValue, _TEdgeWeight>::nec_get_active_count(int *_levels,
                                                                  int _vertices_count,
                                                                  int _desired_level)
{
    int count = 0;
    #pragma _NEC vector
    #pragma omp parallel for reduction(+: count)
    for(int i = 0; i < _vertices_count; i++)
    {
        int val = 0;
        if(_levels[i] == _desired_level)
            val = 1;
        count += val;
    }
    
    return count;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
inline void BFS<_TVertexValue, _TEdgeWeight>::nec_generate_frontier(int *_levels,
                                                                    int *_active_ids,
                                                                    int _vertices_count,
                                                                    int _desired_level,
                                                                    int _threads_count)
{
    int shifts_array[MAX_SX_AURORA_THREADS];
    int pos = 0;
    #pragma omp parallel shared(shifts_array) num_threads(_threads_count)
    {
        int tid = omp_get_thread_num();
        int local_number_of_values = 0;
        
        #pragma omp for schedule(static)
        for(int src_id = 0; src_id < _vertices_count; src_id++)
        {
            if(_levels[src_id] == _desired_level)
            {
                local_number_of_values++;
            }
        }
        
        shifts_array[tid] = local_number_of_values;
        #pragma omp barrier
        
        #pragma omp master
        {
            int cur_shift = 0;
            for(int i = 1; i < _threads_count; i++)
            {
                shifts_array[i] += shifts_array[i - 1];
            }

            for(int i = (_threads_count - 1); i >= 1; i--)
            {
                shifts_array[i] = shifts_array[i - 1];
            }
            shifts_array[0] = 0;
        }
        
        #pragma omp barrier
        
        int tid_shift = shifts_array[tid];
        int *private_ptr = &(_active_ids[tid_shift]);
        
        int local_pos = 0;
        #pragma omp for schedule(static)
        for(int src_id = 0; src_id < _vertices_count; src_id++)
        {
            if(_levels[src_id] == _desired_level)
            {
                private_ptr[local_pos] = src_id;
                local_pos++;
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
inline void BFS<_TVertexValue, _TEdgeWeight>::nec_top_down_step(long long *_outgoing_ptrs,
                                                                int *_outgoing_ids,
                                                                int _vertices_count,
                                                                int _active_count,
                                                                int *_levels,
                                                                int *_active_ids,
                                                                int _cur_level,
                                                                int &_vis,
                                                                int &_in_lvl)
{
    int vis = 0, in_lvl = 0;
    if(_cur_level == 1)
    {
        int src_id = _active_ids[0];
        int src_level = _levels[src_id];
        int connections = _outgoing_ptrs[src_id + 1] - _outgoing_ptrs[src_id];
        int start_pos = _outgoing_ptrs[src_id];
        in_lvl += connections;
        
        if(connections < 2*VECTOR_LENGTH)
        {
            #ifdef __USE_NEC_SX_AURORA__
            #pragma _NEC ivdep
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            #endif
            for(int edge_pos = 0; edge_pos < connections; edge_pos++)
            {
                int dst_id = _outgoing_ids[start_pos + edge_pos];
                if((src_level == _cur_level) && (_levels[dst_id] == -1))
                {
                    _levels[dst_id] = _cur_level + 1;
                    vis++;
                }
            }
        }
        else
        {
            #pragma omp parallel shared(vis)
            {
                int local_vis = 0;
                
                #ifdef __USE_NEC_SX_AURORA__
                #pragma _NEC ivdep
                #pragma _NEC vovertake
                #pragma _NEC novob
                #pragma _NEC vector
                #endif
                #pragma omp for schedule(static)
                for(int edge_pos = 0; edge_pos < connections; edge_pos++)
                {
                    int dst_id = _outgoing_ids[start_pos + edge_pos];
                    int dst_level = 0;
                    //if(dst_id > 5000)
                    dst_level = _levels[dst_id];
                    
                    if((src_level == _cur_level) && (dst_level == -1))
                    {
                        _levels[dst_id] = _cur_level + 1;
                        local_vis++;
                    }
                }
                
                #pragma omp critical
                {
                    vis += local_vis;
                }
            }
        }
    }
    else
    {
        #pragma omp parallel shared(vis, in_lvl)
        {
            int local_in_lvl = 0, local_vis = 0;
            
            int connections[VECTOR_LENGTH];
            int active_reg[VECTOR_LENGTH];
            long long start_pos[VECTOR_LENGTH];
            int vis_reg[VECTOR_LENGTH];
            int in_lvl_reg[VECTOR_LENGTH];
            
            #pragma _NEC vreg(start_pos)
            #pragma _NEC vreg(connections)
            #pragma _NEC vreg(active_reg)
            #pragma _NEC vreg(vis_reg)
            #pragma _NEC vreg(in_lvl_reg)
            
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                connections[i] = 0;
                active_reg[i] = 0;
                start_pos[i] = 0;
                in_lvl_reg[i] = 0;
                vis_reg[i] = 0;
            }
            
            #pragma omp for schedule(static, 1)
            for(int vec_start = 0; vec_start < _active_count; vec_start += VECTOR_LENGTH)
            {
                for(int i = 0; i < VECTOR_LENGTH; i++)
                {
                    active_reg[i] = _active_ids[vec_start + i];
                    
                    int src_id = active_reg[i];
                    connections[i] = _outgoing_ptrs[src_id + 1] - _outgoing_ptrs[src_id];
                    start_pos[i] = _outgoing_ptrs[src_id];
                }
                
                int max_connections = 0;
                for(int i = 0; i < VECTOR_LENGTH; i++)
                {
                    if(max_connections < connections[i])
                        max_connections = connections[i];
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
                        int dst_id = _outgoing_ids[start_pos[i] + edge_pos];
                        
                        in_lvl_reg[i]++;
                        
                        if((i < _active_count) && (edge_pos < connections[i]) && (_levels[src_id] == _cur_level) && (_levels[dst_id] == -1))
                        {
                            _levels[dst_id] = _cur_level + 1;
                            vis_reg[i]++;
                        }
                    }
                }
            }
            
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                local_in_lvl += in_lvl_reg[i];
                local_vis += vis_reg[i];
            }
            
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
inline void BFS<_TVertexValue, _TEdgeWeight>::nec_bottom_up_step(long long *_outgoing_ptrs,
                                                                int *_outgoing_ids,
                                                                int _vertices_count,
                                                                int _active_count,
                                                                int *_levels,
                                                                int *_active_ids,
                                                                int _cur_level,
                                                                int &_vis,
                                                                int &_in_lvl)
{
    int vis = 0;
    long long in_lvl = 0;
    
    if(false)
    {
        int border = 0;
        #pragma omp parallel shared(vis, in_lvl)
        {
            long long local_in_lvl = 0;
            int local_vis = 0;
            
            int connections[VECTOR_LENGTH];
            int active_reg[VECTOR_LENGTH];
            long long start_pos[VECTOR_LENGTH];
            int vis_reg[VECTOR_LENGTH];
            long long in_lvl_reg[VECTOR_LENGTH];
            
            #pragma _NEC vreg(start_pos)
            #pragma _NEC vreg(connections)
            #pragma _NEC vreg(active_reg)
            #pragma _NEC vreg(vis_reg)
            #pragma _NEC vreg(in_lvl_reg)
            
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                connections[i] = 0;
                active_reg[i] = 0;
                start_pos[i] = 0;
                in_lvl_reg[i] = 0;
                vis_reg[i] = 0;
            }
            
            #pragma omp for schedule(static)
            for(int vec_start = 0; vec_start < _active_count; vec_start += VECTOR_LENGTH)
            {
                for(int i = 0; i < VECTOR_LENGTH; i++)
                {
                    active_reg[i] = _active_ids[vec_start + i];
                    
                    int src_id = active_reg[i];
                    connections[i] = _outgoing_ptrs[src_id + 1] - _outgoing_ptrs[src_id];
                    start_pos[i] = _outgoing_ptrs[src_id];
                }
                
                int max_connections = 0;
                for(int i = 0; i < VECTOR_LENGTH; i++)
                {
                    if(max_connections < connections[i])
                        max_connections = connections[i];
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
                        int dst_id = _outgoing_ids[start_pos[i] + edge_pos];
                        
                        if(((vec_start + i) < _active_count) && (edge_pos < connections[i]))
                            in_lvl_reg[i]++;
                         
                        if(((vec_start + i) < _active_count) && (edge_pos < connections[i]) &&
                           (_levels[src_id] == -1) && (_levels[dst_id] == _cur_level))
                        {
                            _levels[src_id] = _cur_level + 1;
                            vis_reg[i]++;
                        }
                    }
                }
            }
            
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                local_in_lvl += in_lvl_reg[i];
                local_vis += vis_reg[i];
            }
            
            #pragma omp critical
            {
                vis += local_vis;
                in_lvl += local_in_lvl;
            }
        }
    }
    else
    {
        #pragma omp parallel shared(vis, in_lvl)
        {
            long long local_in_lvl = 0;
            int local_vis = 0;
            
            int connections[VECTOR_LENGTH];
            int active_reg[VECTOR_LENGTH];
            long long start_pos[VECTOR_LENGTH];
            int vis_reg[VECTOR_LENGTH];
            long long in_lvl_reg[VECTOR_LENGTH];
            
            #pragma _NEC vreg(start_pos)
            #pragma _NEC vreg(connections)
            #pragma _NEC vreg(active_reg)
            #pragma _NEC vreg(vis_reg)
            #pragma _NEC vreg(in_lvl_reg)
            
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                connections[i] = 0;
                active_reg[i] = 0;
                start_pos[i] = 0;
                in_lvl_reg[i] = 0;
                vis_reg[i] = 0;
            }
            
            #pragma omp for schedule(static, 1)
            for(int vec_start = 0; vec_start < _active_count; vec_start += VECTOR_LENGTH)
            {
                for(int i = 0; i < VECTOR_LENGTH; i++)
                {
                    active_reg[i] = _active_ids[vec_start + i];
                    
                    int src_id = active_reg[i];
                    connections[i] = _outgoing_ptrs[src_id + 1] - _outgoing_ptrs[src_id];
                    start_pos[i] = _outgoing_ptrs[src_id];
                }
                
                int max_connections = 0;
                for(int i = 0; i < VECTOR_LENGTH; i++)
                {
                    if(max_connections < connections[i])
                        max_connections = connections[i];
                }
                
                if(max_connections > 4)
                    max_connections = 4;
                
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
                        int dst_id = _outgoing_ids[start_pos[i] + edge_pos];
                        
                        if(((vec_start + i) < _active_count) && (edge_pos < connections[i]))
                            in_lvl_reg[i]++;
                         
                        if(((vec_start + i) < _active_count) && (edge_pos < connections[i]) &&
                           (_levels[src_id] == -1) && (_levels[dst_id] == _cur_level))
                        {
                            _levels[src_id] = _cur_level + 1;
                            vis_reg[i]++;
                        }
                    }
                }
            }
            
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                local_in_lvl += in_lvl_reg[i];
                local_vis += vis_reg[i];
            }
            
            #pragma omp for schedule(static, 1)
            for(int idx = 0; idx < _active_count; idx++)
            {
                int src_id = _active_ids[idx];
                if(_levels[src_id] == -1)
                {
                    int connections = _outgoing_ptrs[src_id + 1] - _outgoing_ptrs[src_id];
                    long long start_pos = _outgoing_ptrs[src_id];
                    
                    #pragma _NEC ivdep
                    #pragma _NEC vovertake
                    #pragma _NEC novob
                    #pragma _NEC vector
                    for(int edge_pos = 0; edge_pos < connections; edge_pos ++)
                    {
                        int dst_id = _outgoing_ids[start_pos + edge_pos];
                        
                        local_in_lvl ++;
                        if((_levels[src_id] == -1) && (_levels[dst_id] == _cur_level))
                        {
                            _levels[src_id] = _cur_level + 1;
                            local_vis++;
                        }
                    }
                }
            }
            
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

void remove_zero_nodes(long long *_outgoing_ptrs, int _vertices_count, int *_levels)
{
    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma omp parallel for schedule(static, 1)
    for(int src_id = 0; src_id < _vertices_count; src_id++)
    {
        int connections = _outgoing_ptrs[src_id + 1] - _outgoing_ptrs[src_id];
        if(connections == 0)
        {
            _levels[src_id] = -2;
        }
    }
}

void mark_zero_nodes(int _vertices_count, int *_levels)
{
    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma omp parallel for schedule(static, 1)
    for(int src_id = 0; src_id < _vertices_count; src_id++)
    {
        if(_levels[src_id] == -2)
        {
            _levels[src_id] = -1;
        }
    }
}

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
    _TEdgeWeight *outgoing_weights = _graph.get_outgoing_weights();
    int threads_count = omp_get_max_threads();
    
    cout << "1!!! " << ((double)(outgoing_ptrs[1] - outgoing_ptrs[0]))/((double)edges_count) << endl << endl;
    
    int *active_ids = new int[vertices_count];
    
    #ifdef __USE_NEC_SX_AURORA__
    #pragma _NEC retain(_levels)
    #endif
    
    #pragma omp parallel for
    for(int i = 0; i < vertices_count; i++)
    {
        _levels[i] = -1;
        active_ids[i] = 0;
    }
    
    int cur_level = 1;
    _levels[_source_vertex] = cur_level;
    int active_count = 1;
    active_ids[0] = _source_vertex;
    
    StateOfBFS current_state = TOP_DOWN;
    double total_time = 0;
    
    t1 = omp_get_wtime();
    remove_zero_nodes(outgoing_ptrs, vertices_count, _levels);
    t2 = omp_get_wtime();
    //total_time += t2 - t1;
    
    while(active_count > 0)
    {
        int vis = 0, in_lvl = 0;
        int current_active_count = active_count;
        
        t3 = omp_get_wtime();
        if(current_state == TOP_DOWN)
        {
            nec_top_down_step(outgoing_ptrs, outgoing_ids, vertices_count, active_count, _levels, active_ids, cur_level,
                              vis, in_lvl);
        }
        else if(current_state == BOTTOM_UP)
        {
            nec_bottom_up_step(outgoing_ptrs, outgoing_ids, vertices_count, active_count, _levels, active_ids, cur_level,
                               vis, in_lvl);
        }
        
        t4 = omp_get_wtime();
        total_time += t4 - t3;
        double kernel_time = t4 - t3;
        
        t3 = omp_get_wtime();
        int next_active_count = nec_get_active_count(_levels, vertices_count, cur_level + 1);
        int frontier_size = next_active_count;
        
        StateOfBFS next_state = change_state(current_active_count, next_active_count, vertices_count, edges_count, current_state,
                                             vis, in_lvl);
        
        if(next_state == TOP_DOWN)
        {
            nec_generate_frontier(_levels, active_ids, vertices_count, cur_level + 1, threads_count);
            active_count = next_active_count;
        }
        else if(next_state == BOTTOM_UP)
        {
            nec_generate_frontier(_levels, active_ids, vertices_count, -1, threads_count);
            active_count = nec_get_active_count(_levels, vertices_count, -1);
        }
        t4 = omp_get_wtime();
        double reminder_time = t4 - t3;
        total_time += reminder_time;
        
        if(current_state == TOP_DOWN)
            cout << "level " << cur_level << " in TD state" << endl;
        else if(current_state == BOTTOM_UP)
            cout << "level " << cur_level << " in BU state" << endl;
        //cout << "front size: " << 100.0 * ((double)frontier_size)/ ((double)vertices_count) << " %" << endl;
        cout << "ALL GRAPH PERF: " << ((double)edges_count)/(kernel_time*1e6) << " MTEPS" << endl;
        cout << "REAL perf: " << ((double)in_lvl)/(kernel_time*1e6) << " MTEPS" << endl;
        cout << "kernel band: " << (3.0 * sizeof(int))*((double)in_lvl)/(kernel_time*1e9) << " GB/s" << endl;
        //cout << "reminder perf: " << ((double)edges_count)/(reminder_time*1e6) << " MTEPS" << endl;
        cout << endl;
        
        current_state = next_state;
        cur_level++;
    }
    
    t1 = omp_get_wtime();
    mark_zero_nodes(vertices_count, _levels);
    t2 = omp_get_wtime();
    //total_time += t2 - t1;
    
    cout << "TOTAL BFS Perf: " << ((double)edges_count)/(total_time*1e6) << " MTEPS" << endl << endl << endl;
    
    delete []active_ids;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* nec_bfs_h */
