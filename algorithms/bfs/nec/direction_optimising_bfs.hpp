//
//  direction_optimising_bfs.hpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 30/06/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef nec_direction_optimising_bfs_hpp
#define nec_direction_optimising_bfs_hpp

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void BFS<_TVertexValue, _TEdgeWeight>::nec_top_down_step(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                        int *_outgoing_ids,
                                                        int _number_of_vertices_in_first_part,
                                                        int *_levels,
                                                        VertexQueue &_global_queue,
                                                        VertexQueue **_local_queues,
                                                        int _omp_threads,
                                                        int _current_level,
                                                        int &_vis,
                                                        int &_in_lvl)
{
    int src_id = _global_queue.get_data()[0];
            
    long long edge_start = _graph.get_vertex_pointer(src_id);
    int connections_count = _graph.get_vector_connections_count(src_id);
    
    #pragma omp parallel num_threads(_omp_threads)
    {
        int local_vis_reg[VECTOR_LENGTH];
        
        #ifdef __USE_NEC_SX_AURORA__
        #pragma _NEC vreg(local_vis_reg)
        #pragma _NEC retain(_levels)
        #endif
        
        int local_vis = 0, local_in_lvl = 0;
        
        #pragma unroll
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            local_vis_reg[i] = 0;
        }
        
        #pragma omp for schedule(static)
        for(int edge_pos = 0; edge_pos < connections_count; edge_pos += VECTOR_LENGTH)
        {
            local_in_lvl += VECTOR_LENGTH;
            
            #ifdef __USE_NEC_SX_AURORA__
            #pragma simd
            #pragma _NEC ivdep
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma unroll
            #endif
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                int dst_id = _outgoing_ids[edge_start + edge_pos + i];
                
                if(_levels[dst_id] == -1)
                {
                    _levels[dst_id] = _current_level;
                    local_vis_reg[i]++;
                }
            }
        }
        
        #pragma unroll
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            local_vis += local_vis_reg[i];
        }
        
        #pragma omp atomic
        _vis += local_vis;
        
        #pragma omp atomic
        _in_lvl += local_in_lvl;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void BFS<_TVertexValue, _TEdgeWeight>::nec_bottom_up_step(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                          long long *_first_part_ptrs,
                                                          int *_first_part_sizes,
                                                          int _vector_segments_count,
                                                          long long *_vector_group_ptrs,
                                                          int *_vector_group_sizes,
                                                          int *_outgoing_ids,
                                                          int _vertices_count,
                                                          int _number_of_vertices_in_first_part,
                                                          int *_levels,
                                                          int _omp_threads,
                                                          int _current_level,
                                                          int &_vis,
                                                          int &_in_lvl)
{
    #pragma omp parallel num_threads(_omp_threads)
    {
        int local_vis = 0, local_in_lvl = 0;
        
        int reg_levels[VECTOR_LENGTH];
        int break_flags_reg[VECTOR_LENGTH];
        int local_vis_reg[VECTOR_LENGTH];
        
        #ifdef __USE_NEC_SX_AURORA__
        #pragma _NEC vreg(reg_levels)
        #pragma _NEC vreg(break_flags_reg)
        #pragma _NEC vreg(local_vis_reg)
        #endif
        
        #ifdef __USE_NEC_SX_AURORA__
        #pragma _NEC vector
        #endif
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            reg_levels[i] = 0;
            break_flags_reg[i] = 0;
            local_vis_reg[i] = 0;
        }
        
        /*#pragma omp for schedule(static, 1)
        for(int src_id = 0; src_id < _number_of_vertices_in_first_part; src_id++)
        {
            long long edge_start = _first_part_ptrs[src_id];
            int connections_count = _first_part_sizes[src_id];
            
            if(_levels[src_id] == -1)
            {
                local_in_lvl += connections_count;
                
                for(int edge_pos = 0; edge_pos < connections_count; edge_pos++)
                {
                    local_in_lvl++;
                    int dst_id = _outgoing_ids[edge_start + edge_pos];
                    if(_levels[dst_id] == (_current_level - 1))
                    {
                        _levels[src_id] = _current_level;
                        local_vis++;
                        break;
                    }
                }
            }
        }*/
        
        #pragma omp for schedule(static, 1)
        for(int segment_first_vertex = _number_of_vertices_in_first_part; segment_first_vertex < _vertices_count; segment_first_vertex += VECTOR_LENGTH)
        {
            #ifdef __USE_NEC_SX_AURORA__
            #pragma _NEC vector
            #endif
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                reg_levels[i] = _levels[segment_first_vertex + i];
                if(reg_levels[i] >= 0)
                    break_flags_reg[i] = 1;
                else
                    break_flags_reg[i] = 0;
            }
            
            int cur_vector_segment = segment_first_vertex / VECTOR_LENGTH;
            long long segement_edges_start = _vector_group_ptrs[cur_vector_segment];
            int segment_connections_count  = _vector_group_sizes[cur_vector_segment];
            
            for(int edge_pos = 0; edge_pos < segment_connections_count; edge_pos++)
            {
                local_in_lvl += VECTOR_LENGTH;
                
                #ifdef __USE_NEC_SX_AURORA__
                #pragma simd
                #pragma _NEC ivdep
                #pragma _NEC vovertake
                #pragma _NEC vob
                #pragma _NEC vector
                #endif
                for(int i = 0; i < VECTOR_LENGTH; i++)
                {
                    int src_level = reg_levels[i];
                    
                    int src_id = segment_first_vertex + i;
                    int dst_id = _outgoing_ids[segement_edges_start + edge_pos * VECTOR_LENGTH + i];
                    
                    int dst_level = _levels[dst_id];
                    
                    if((src_level == -1) && (dst_level == (_current_level - 1)))
                    {
                        reg_levels[i] = _current_level;
                        break_flags_reg[i] = 1;
                        local_vis_reg[i]++;
                    }
                }

                int break_flag = 0;
                for(int i = 0; i < VECTOR_LENGTH; i++)
                    break_flag += break_flags_reg[i];
                
                if((break_flag >= VECTOR_LENGTH))
                    break;
            }
            
            #pragma simd
            #pragma unroll
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                _levels[segment_first_vertex + i] = reg_levels[i];
            }
        }
        
        #pragma unroll
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            local_vis += local_vis_reg[i];
        }
        
        #pragma omp barrier
        
        #pragma omp atomic
        _vis += local_vis;
        
        #pragma omp atomic
        _in_lvl += local_in_lvl;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void BFS<_TVertexValue, _TEdgeWeight>::nec_direction_optimising_BFS(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                                    int *_levels,
                                                                    int _source_vertex)
{
    /*LOAD_VECTORISED_CSR_GRAPH_DATA(_graph)
    
    int omp_threads = omp_get_max_threads();
    _graph.set_threads_count(omp_threads);
    
    VertexQueue **local_queues = new VertexQueue *[omp_threads];
    for(int i = 0; i < omp_threads; i++)
        local_queues[i] = new VertexQueue(vertices_count);
    
    _graph.template vertex_array_set_to_constant<int>(_levels, -1);
    _graph.template vertex_array_set_element<int>(_levels, _source_vertex, 1);
    
    VertexQueue global_queue(vertices_count);
    global_queue.push_back(_source_vertex);
    int current_level = 2;
    
    vector<int> level_num;
    vector<double> level_perf;
    vector<string> level_state;
    
    //ftrace_region_begin("bfs_inv");
    
    double t1 = omp_get_wtime();
    StateOfBFS current_state = BOTTOM_UP;
    //while(!global_queue.empty())
    while(true)
    //while(current_level < 10)
    {
        double t3 = omp_get_wtime();
        
        int current_queue_size = global_queue.get_size();
        int vis = 0, in_lvl = 0;
        
        if(current_level == 2)
        {
            nec_top_down_step(_graph, outgoing_ids, number_of_vertices_in_first_part, _levels, global_queue, local_queues,
                                omp_threads, current_level, vis, in_lvl);
        }
        else
        {
            nec_bottom_up_step(_graph, first_part_ptrs, first_part_sizes, vector_segments_count, vector_group_ptrs,
                           vector_group_sizes, outgoing_ids, vertices_count, number_of_vertices_in_first_part,
                           _levels, global_queue, local_queues, omp_threads, current_level, vis, in_lvl);
        }
        
        double t4 = omp_get_wtime();
        
        double real_bandwidth = (2.0 * sizeof(int) * in_lvl) / ((t4 - t3) * 1e9);
        double global_bandwidth = (2.0 * sizeof(int) * edges_count) / ((t4 - t3) * 1e9);
        cout << in_lvl << " vs " << edges_count << endl;
        cout << "global bandwidth: " << global_bandwidth << " GB/s" << endl;
        cout << "real bandwidth: " << real_bandwidth << " GB/s" << endl;
        cout << (double)edges_count / (double)in_lvl << " and " << global_bandwidth / real_bandwidth << endl << endl;
        
        current_level++;
        
        int next_queue_size = global_queue.get_size();
        
        current_state = change_state(current_queue_size, next_queue_size, vertices_count, edges_count,
                                     current_state, vis, in_lvl);
        
        cout << vis << endl;
        if(vis == 0)
            break;
        
        level_num.push_back(current_level - 1);
        level_perf.push_back(t4 - t3);
        level_state.push_back("BU");
    }
    double t2 = omp_get_wtime();
    
    //ftrace_region_end("bfs_inv");
    
    cout << "BFS perf: " << ((double)edges_count) / ((t2 - t1) * 1e6) << " MFLOPS" << endl;
    for(int i = 0; i < level_perf.size(); i++)
        cout << "level " << level_num[i] << " in " << level_state[i] <<  " | perf: " << ((double)edges_count) / (level_perf[i] * 1e6) << " MFLOPS | " << level_perf[i]*1000.0 << " ms" << endl;
    
    for(int i = 0; i < omp_threads; i++)
        delete local_queues[i];
    delete []local_queues;*/
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
int BFS<_TVertexValue, _TEdgeWeight>::number_of_active(int *_levels, int _vertices_count, int _current_level)
{
    int sum = 0;
    /*#pragma omp parallel for shared(sum) reduction(+: sum)
    for (auto i = 0; i < _vertices_count; i++)
    {
        int data_val = 0;
        if(_levels[i] == _current_level)
            data_val = 1;
        sum += data_val;
    }*/
    
    return sum;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void BFS<_TVertexValue, _TEdgeWeight>::test_primitives(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                       int *_levels,
                                                       int _source_vertex)
{
    int vertices_count    = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count   ();
    long long    *outgoing_ptrs    = _graph.get_outgoing_ptrs   ();
    int          *outgoing_ids     = _graph.get_outgoing_ids    ();
    _TEdgeWeight *outgoing_weights = _graph.get_outgoing_weights();
    int threads_count = omp_get_max_threads();
    
    #ifdef __USE_NEC_SX_AURORA__
    #pragma _NEC retain(_levels)
    #endif
    
    int changes = 1;
    int current_iteration = 1;
    double t1 = omp_get_wtime();
    #pragma omp parallel num_threads(8)
    {
        int reg_levels[VECTOR_LENGTH];
        int reg_connections[VECTOR_LENGTH];
        int reg_starts[VECTOR_LENGTH];
        #ifdef __USE_NEC_SX_AURORA__
        #pragma _NEC vreg(reg_levels)
        #endif
        
        #ifdef __USE_NEC_SX_AURORA__
        #pragma _NEC vector
        #endif
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            reg_levels[i] = 0;
        }
        
        #pragma omp for schedule(static)
        for(int src_id = 0; src_id < vertices_count; src_id += 4)
        {
            long long edge_start = outgoing_ptrs[src_id];
            int connections_count = outgoing_ptrs[src_id + 1] - outgoing_ptrs[src_id];
            int src_val = _levels[src_id];
            
            for(int edge_pos = 0; edge_pos < connections_count; edge_pos += 256)
            {
                #ifdef __USE_NEC_SX_AURORA__
                #pragma _NEC ivdep
                #pragma _NEC vovertake
                #pragma _NEC novob
                #pragma _NEC vector
                #endif
                for(int i = 0; i < 256; i++)
                {
                    if((edge_pos + i) < connections_count)
                    {
                        int dst_id = outgoing_ids[edge_start + edge_pos + i];
                        int dst_val = _levels[dst_id];
                        
                        if(src_val < dst_val)
                        {
                            reg_levels[i] = dst_val;
                        }
                    }
                }
            }
            
            int max = 0;
            #ifdef __USE_NEC_SX_AURORA__
            #pragma _NEC vector
            #endif
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                max += reg_levels[i];
            }
            
            _levels[src_id] = max;
        }
    }
    double t2 = omp_get_wtime();
    
    cout << "BFS time: " << t2 - t1 << endl;
    cout << "BFS Perf: " << ((double)edges_count) / ((t2 - t1) * 1e6) << " MFLOPS" << endl;
    cout << "BFS bandwidth: " << ((double)current_iteration)*((double)edges_count * (4*sizeof(int))) / ((t2 - t1) * 1e9) << " GB/s" << endl << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* nec_direction_optimising_bfs_hpp */
