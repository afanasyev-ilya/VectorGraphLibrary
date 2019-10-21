//
//  bfs.hpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 03/06/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef bfs_hpp
#define bfs_hpp

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void BFS<_TVertexValue, _TEdgeWeight>::allocate_result_memory(int _vertices_count, int **_levels)
{
    *_levels = new int[_vertices_count];
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void BFS<_TVertexValue, _TEdgeWeight>::free_result_memory(int *_levels)
{
    delete[] _levels;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _TVertexValue, typename _TEdgeWeight>
void BFS<_TVertexValue, _TEdgeWeight>::gpu_direction_optimising_BFS(
                                         VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                         int *_levels,
                                         int _source_vertex)
{
    LOAD_VECTORISED_CSR_GRAPH_DATA(_graph)
    
    cout << "gpu bfs test" << endl;
    
    int *device_levels;
    SAFE_CALL(cudaMalloc((void**)&device_levels, vertices_count * sizeof(int)));
    
    gpu_direction_optimising_bfs_wrapper(first_part_ptrs, first_part_sizes,
                                         vector_segments_count,
                                         vector_group_ptrs, vector_group_sizes,
                                         outgoing_ids, number_of_vertices_in_first_part,
                                         device_levels, vertices_count, edges_count, _source_vertex);
    
    SAFE_CALL(cudaMemcpy(_levels, device_levels, vertices_count * sizeof(int), cudaMemcpyDeviceToHost));
    
    SAFE_CALL(cudaFree(device_levels));
}

#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline int get_active_count(int *_levels, int _vertices_count, int _desired_level)
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

inline void generate_frontier(int *_levels, int *_active_ids, int _vertices_count, int _desired_level, int _threads_count)
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

inline void top_down_step(long long *_outgoing_ptrs,
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
        int connections = _outgoing_ptrs[src_id + 1] - _outgoing_ptrs[src_id];
        int start_pos = _outgoing_ptrs[src_id];
        in_lvl += connections;
        
        #ifdef __USE_NEC_SX_AURORA__
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vector
        #endif
        for(int edge_pos = 0; edge_pos < connections; edge_pos++)
        {
            int dst_id = _outgoing_ids[start_pos + edge_pos];
            if((_levels[src_id] == _cur_level) && (_levels[dst_id] == -1))
            {
                _levels[dst_id] = _cur_level + 1;
                vis++;
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

inline void bottom_up_step(long long *_outgoing_ptrs,
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
    
    if(true)
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
                         
                        if((i < _vertices_count) && (edge_pos < connections[i]) &&
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
    _vis = vis;
    _in_lvl = in_lvl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void BFS<_TVertexValue, _TEdgeWeight>::new_bfs(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                               int *_levels,
                                               int _source_vertex)
{
    int vertices_count    = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count   ();
    long long    *outgoing_ptrs    = _graph.get_outgoing_ptrs   ();
    int          *outgoing_ids     = _graph.get_outgoing_ids    ();
    _TEdgeWeight *outgoing_weights = _graph.get_outgoing_weights();
    int threads_count = omp_get_max_threads();
    
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
    while(active_count > 0)
    {
        int vis = 0, in_lvl = 0;
        cout << "current state: " << current_state << endl;
        
        int current_active_count = active_count;
        
        double t3 = omp_get_wtime();
        if(current_state == TOP_DOWN)
        {
            top_down_step(outgoing_ptrs, outgoing_ids, vertices_count, active_count, _levels, active_ids, cur_level, vis, in_lvl);
        }
        else if(current_state == BOTTOM_UP)
        {
            bottom_up_step(outgoing_ptrs, outgoing_ids, vertices_count, active_count, _levels, active_ids, cur_level, vis, in_lvl);
        }
        
        double t4 = omp_get_wtime();
        total_time += t4 - t3;
        double kernel_time = t4 - t3;
        
        t3 = omp_get_wtime();
        active_count = get_active_count(_levels, vertices_count, cur_level + 1);
        
        int next_active_count = active_count;
        current_state = change_state(current_active_count, next_active_count, vertices_count, edges_count, current_state,
                                     vis, in_lvl);
        
        if(current_state == TOP_DOWN)
        {
            generate_frontier(_levels, active_ids, vertices_count, cur_level + 1, threads_count);
        }
        else
        {
            generate_frontier(_levels, active_ids, vertices_count, -1, threads_count);
            active_count = get_active_count(_levels, vertices_count, -1);
        }
        t4 = omp_get_wtime();
        double reminder_time = t4 - t3;
        total_time += reminder_time;
        
        cout << "level " << cur_level << " perf: " << ((double)edges_count)/(kernel_time*1e6) << " MTEPS" << endl;
        cout << "reminder " << cur_level << " perf: " << ((double)edges_count)/(reminder_time*1e6) << " MTEPS" << endl;
        cout << "kernel band: " << (3.0 * sizeof(int))*((double)in_lvl)/(kernel_time*1e9) << " GB/s" << endl << endl;
        
        cur_level++;
    }
    
    double t2 = omp_get_wtime();
    cout << "BFS Perf: " << ((double)edges_count)/(total_time*1e6) << " MTEPS" << endl;
    
    delete []active_ids;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* bfs_hpp */
