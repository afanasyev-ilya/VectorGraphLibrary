#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
template <typename _TVertexValue, typename _TEdgeWeight>
void ConnectedComponents<_TVertexValue,_TEdgeWeight>::nec_shiloach_vishkin(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int *_vertices_data)
{
    LOAD_VECTORISED_CSR_GRAPH_REVERSE_DATA(_graph)
    
    int threads_count = omp_get_max_threads();
    _graph.set_threads_count(threads_count);
    
    #ifdef __USE_NEC_SX_AURORA__
    #pragma _NEC retain(_vertices_data)
    #endif
    
    #pragma omp parallel for num_threads(threads_count)
    for(int i = 0; i < vertices_count; i++)
    {
        _vertices_data[i] = i;
    }
    
    int *cached_vertices_data = _graph.template allocate_private_caches<int>(threads_count);
    
    int hook_changes = 1, jump_changes = 1;
    int current_iteration = 1;
    double t1 = omp_get_wtime();
    #pragma omp parallel num_threads(threads_count) shared(hook_changes, jump_changes)
    {
        int reg_hook_changes[VECTOR_LENGTH];
        int reg_jump_changes[VECTOR_LENGTH];
        #ifdef __USE_NEC_SX_AURORA__
        #pragma _NEC vreg(reg_hook_changes)
        #pragma _NEC vreg(reg_jump_changes)
        #endif
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            reg_hook_changes[i] = 0;
            reg_jump_changes[i] = 0;
        }
        
        int thread_id = omp_get_thread_num();
        int *private_vertices_data = &cached_vertices_data[thread_id * CACHED_VERTICES * CACHE_STEP];
        
        while(hook_changes)
        {
            #pragma omp barrier
            
            _graph.template place_data_into_cache<int>(_vertices_data, private_vertices_data);
            
            #pragma omp barrier
            
            hook_changes = 0;
            #ifdef __USE_NEC_SX_AURORA__
            #pragma _NEC vector
            #endif
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                reg_hook_changes[i] = 0;
            }
            
            if(number_of_vertices_in_first_part > 0)
            {
                for(int src_id = 0; src_id < number_of_vertices_in_first_part; src_id++)
                {
                    long long edge_start = first_part_ptrs[src_id];
                    int connections_count = first_part_sizes[src_id];
                    
                    #pragma omp for schedule(static, 1)
                    for(long long edge_pos = 0; edge_pos < connections_count; edge_pos += VECTOR_LENGTH)
                    {
                        #ifdef __USE_NEC_SX_AURORA__
                        #pragma _NEC ivdep
                        #pragma _NEC vovertake
                        #pragma _NEC novob
                        #pragma _NEC vector
                        #endif
                        for(int i = 0; i < VECTOR_LENGTH; i++)
                        {
                            int dst_id = incoming_ids[edge_start + edge_pos + i];
                            
                            int src_val = _graph.template load_vertex_data<int>(src_id, _vertices_data);
                            int dst_val = _graph.template load_vertex_data_cached<int>(dst_id, _vertices_data, private_vertices_data);
                            
                            if(src_val < dst_val)
                            {
                                int dst_dst_val = _graph.template load_vertex_data_cached<int>(dst_val, _vertices_data, private_vertices_data);
                                if(dst_val == dst_dst_val)
                                {
                                    _vertices_data[dst_val] = src_val;
                                    reg_hook_changes[i] = 1;
                                }
                            }
                        }
                    }
                }
            }
            
            #pragma omp for schedule(static, 1)
            for(int cur_vector_segment = 0; cur_vector_segment < vector_segments_count; cur_vector_segment++)
            {
                int segment_first_vertex = cur_vector_segment * VECTOR_LENGTH + number_of_vertices_in_first_part;
                
                long long segement_edges_start = vector_group_ptrs[cur_vector_segment];
                int segment_connections_count  = vector_group_sizes[cur_vector_segment];
                
                for(long long edge_pos = 0; edge_pos < segment_connections_count; edge_pos++)
                {
                    #ifdef __USE_NEC_SX_AURORA__
                    #pragma _NEC ivdep
                    #pragma _NEC vovertake
                    #pragma _NEC novob
                    #pragma _NEC vector
                    #endif
                    for(int i = 0; i < VECTOR_LENGTH; i++)
                    {
                        int src_id = segment_first_vertex + i;
                        int dst_id = incoming_ids[segement_edges_start + edge_pos * VECTOR_LENGTH + i];
                        
                        int src_val = _graph.template load_vertex_data<int>(src_id, _vertices_data);
                        int dst_val = _graph.template load_vertex_data_cached<int>(dst_id, _vertices_data, private_vertices_data);
                        
                        if(src_val < dst_val)
                        {
                            int dst_dst_val = _graph.template load_vertex_data_cached<int>(dst_val, _vertices_data, private_vertices_data);
                            if(dst_val == dst_dst_val)
                            {
                                _vertices_data[dst_val] = src_val;
                                reg_hook_changes[i] = 1;
                            }
                        }
                    }
                }
            }
            
            #pragma omp barrier
            
            int private_hook_changes = 0;
            #ifdef __USE_NEC_SX_AURORA__
            #pragma _NEC vector
            #endif
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                private_hook_changes += reg_hook_changes[i];
            }
            
            #pragma omp atomic
            hook_changes += private_hook_changes;
            
            _graph.template place_data_into_cache<int>(_vertices_data, private_vertices_data);
            
            #pragma omp barrier
            
            while(jump_changes)
            {
                #pragma omp barrier
                
                jump_changes = 0;
                
                #pragma omp barrier
                
                #ifdef __USE_NEC_SX_AURORA__
                #pragma _NEC vector
                #endif
                for(int i = 0; i < VECTOR_LENGTH; i++)
                {
                    reg_jump_changes[i] = 0;
                }
                
                #pragma omp for schedule(static, 1)
                for(int pos = 0; pos < vertices_count; pos += VECTOR_LENGTH)
                {
                    #ifdef __USE_NEC_SX_AURORA__
                    #pragma _NEC ivdep
                    #pragma _NEC vovertake
                    #pragma _NEC novob
                    #pragma _NEC vector
                    #endif
                    for(int i = 0; i < VECTOR_LENGTH; i++)
                    {
                        int src_id = pos + i;
                        if(src_id < vertices_count)
                        {
                            int src_val = _graph.template load_vertex_data<int>(src_id, _vertices_data);
                            int src_src_val = _graph.template load_vertex_data_cached<int>(src_val, _vertices_data, private_vertices_data);
                            
                            if(src_val != src_src_val)
                            {
                                _vertices_data[src_id] = src_src_val;
                                reg_jump_changes[i]++;
                            }
                        }
                    }
                }
                
                int private_jump_changes = 0;
                
                #ifdef __USE_NEC_SX_AURORA__
                #pragma _NEC vector
                #endif
                for(int i = 0; i < VECTOR_LENGTH; i++)
                {
                    private_jump_changes += reg_jump_changes[i];
                }
                jump_changes += private_jump_changes;
                
            #pragma omp barrier
            }
            
            #pragma omp master
            {
                current_iteration++;
            }
            
            #pragma omp barrier
        }
    }
    double t2 = omp_get_wtime();
    
    cout << "CC time: " << t2 - t1 << endl;
    cout << "CC Perf: " << ((double)edges_count) / ((t2 - t1) * 1e6) << " MFLOPS" << endl;
    cout << "CC iterations count: " << current_iteration << endl;
    cout << "CC Perf per iteration: " << current_iteration * ((double)edges_count) / ((t2 - t1) * 1e6) << " MFLOPS" << endl;
    cout << "CC bandwidth: " << ((double)current_iteration)*((double)edges_count * (5*sizeof(int))) / ((t2 - t1) * 1e9) << " GB/s" << endl << endl;
    
    _graph.template free_data<int>(cached_vertices_data);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ConnectedComponents<_TVertexValue,_TEdgeWeight>::nec_shiloach_vishkin(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int *_cc_result)
{
    int vertices_count    = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count   ();
    long long    *outgoing_ptrs    = _graph.get_outgoing_ptrs   ();
    int          *outgoing_ids     = _graph.get_outgoing_ids    ();
    _TEdgeWeight *outgoing_weights = _graph.get_outgoing_weights();
    int threads_count = omp_get_max_threads();
    
    #ifdef __USE_NEC_SX_AURORA__
    #pragma _NEC retain(_cc_result)
    #endif
    
    #pragma omp parallel for num_threads(threads_count)
    for(int i = 0; i < vertices_count; i++)
    {
        _cc_result[i] = i;
    }
    
    int hook_changes = 1;
    int current_iteration = 1;
    while(hook_changes)
    {
        // hook
        hook_changes = 0;
        #pragma omp parallel for schedule(static) shared(hook_changes)
        for(int src_id = 0; src_id < vertices_count; src_id++)
        {
            long long edge_start = outgoing_ptrs[src_id];
            int connections_count = outgoing_ptrs[src_id + 1] - outgoing_ptrs[src_id];
            
            for(int edge_pos = 0; edge_pos < connections_count; edge_pos++)
            {
                int dst_id = outgoing_ids[edge_start + edge_pos];
                int src_val = _cc_result[src_id];
                int dst_val = _cc_result[dst_id];
                
                if((src_val < dst_val) && (dst_val == _cc_result[dst_val]))
                {
                    _cc_result[dst_val] = src_val;
                    hook_changes = true;
                }
            }
        }
        
        // jump
        #pragma omp parallel for schedule(static)
        for(int src_id = 0; src_id < vertices_count; src_id++)
        {
            while(_cc_result[src_id] != _cc_result[_cc_result[src_id]])
            {
                _cc_result[src_id] = _cc_result[_cc_result[src_id]];
            }
        }
        
        current_iteration++;
    }
}
*/
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
