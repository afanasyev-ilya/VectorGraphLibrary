#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _TVertexValue, typename _TEdgeWeight>
void ConnectedComponents<_TVertexValue,_TEdgeWeight>::nec_shiloach_vishkin(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                                           int *_components)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    GraphPrimitivesNEC graph_API;
    FrontierNEC frontier(_graph.get_vertices_count());

    auto init_components_op = [_components] (int src_id)
    {
        _components[src_id] = src_id;
    };
    graph_API.compute(init_components_op, vertices_count);

    auto all_active = [] (int src_id)->int
    {
        return NEC_IN_FRONTIER_FLAG;
    };
    frontier.filter(_graph, all_active);

    int hook_changes = 1, jump_changes = 1;

    while(hook_changes)
    {
        double t1 = omp_get_wtime();
        #pragma omp parallel
        {
            hook_changes = 0;
            NEC_REGISTER_INT(hook_changes, 0);

            auto edge_op = [_components, &reg_hook_changes](int src_id, int dst_id, int local_edge_pos,
                    long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
            {
                int src_val = _components[src_id];
                int dst_val = _components[dst_id];
                int dst_dst_val = _components[dst_val];
                if((src_val < dst_val) && (dst_val == dst_dst_val))
                {
                    _components[dst_val] = src_val;
                    reg_hook_changes[vector_index] = 1;
                }
            };

            graph_API.advance(_graph, frontier, edge_op);

            #pragma omp atomic
            hook_changes += register_sum_reduce(reg_hook_changes);
        }
        double t2 = omp_get_wtime();
        cout << "time: " << (t2 - t1)*1000.0 << " ms" << endl;
        cout << "hook changes perf: " << edges_count / ((t2 - t1)*1e6) << " MTEPS" << endl;

        jump_changes = 1;
        while(jump_changes)
        {
            jump_changes = 0;
            NEC_REGISTER_INT(jump_changes, 0);

            auto jump_op = [_components, &reg_jump_changes](int src_id)
            {
                int src_val = _components[src_id];
                int src_src_val = _components[src_val];
                int vector_index = src_id % VECTOR_LENGTH;

                if(src_val != src_src_val)
                {
                    _components[src_id] = src_src_val;
                    reg_jump_changes[vector_index]++;
                }
            };

            graph_API.compute(jump_op, vertices_count);

            jump_changes += register_sum_reduce(reg_jump_changes);
        }

        cout << "iter done " << hook_changes << endl;
    }

    print_component_stats(_components, vertices_count);
}

#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ConnectedComponents<_TVertexValue,_TEdgeWeight>::test_shiloach_vishkin(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int *_vertices_data)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);
    
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

            const int number_of_vertices_in_first_part = _graph.get_nec_vector_core_threshold_vertex();

            if(number_of_vertices_in_first_part > 0)
            {
                for(int src_id = 0; src_id < number_of_vertices_in_first_part; src_id++)
                {
                    long long edge_start = outgoing_ptrs[src_id];
                    int connections_count = outgoing_ptrs[src_id + 1] - outgoing_ptrs[src_id];
                    
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
                            int dst_id = outgoing_ids[edge_start + edge_pos + i];
                            
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
            for(int cur_vector_segment = 0; cur_vector_segment < ve_vector_segments_count; cur_vector_segment++)
            {
                int segment_first_vertex = cur_vector_segment * VECTOR_LENGTH + number_of_vertices_in_first_part;
                
                long long segement_edges_start = ve_vector_group_ptrs[cur_vector_segment];
                int segment_connections_count  = ve_vector_group_sizes[cur_vector_segment];
                
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
                        int dst_id = ve_adjacent_ids[segement_edges_start + edge_pos * VECTOR_LENGTH + i];
                        
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

    print_component_stats(_vertices_data, vertices_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
