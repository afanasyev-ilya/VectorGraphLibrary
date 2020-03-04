//
//  bellman_ford.hpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 08/09/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef bellman_ford_hpp
#define bellman_ford_hpp

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void WidestPaths<_TVertexValue, _TEdgeWeight>::bellman_ford(
                                      VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_reversed_graph,
                                      int _source_vertex, _TEdgeWeight *_widths)
{
    LOAD_VECTORISED_CSR_GRAPH_REVERSE_DATA(_reversed_graph)
    
    int threads_count = omp_get_max_threads();
    _reversed_graph.set_threads_count(threads_count);
    
    #ifdef __USE_NEC_SX_AURORA__
    #pragma _NEC retain(_widths)
    #endif
    
    _reversed_graph.template vertex_array_set_to_constant<_TEdgeWeight>(_widths, 0.0);
    _reversed_graph.template vertex_array_set_element<_TEdgeWeight>(_widths, _source_vertex, FLT_MAX);
    
    _TEdgeWeight *cached_widths = _reversed_graph.template allocate_private_caches<_TEdgeWeight>(threads_count);
    
    double t1 = omp_get_wtime();
    int changes = 1;
    int iterations_count = 0;
    #pragma omp parallel num_threads(threads_count) shared(changes)
    {
        int reg_changes[VECTOR_LENGTH];
        _TEdgeWeight reg_widths[VECTOR_LENGTH];
        
        #ifdef __USE_NEC_SX_AURORA__
        #pragma _NEC vreg(reg_changes)
        #pragma _NEC vreg(reg_widths)
        #endif
        
        #ifdef __USE_NEC_SX_AURORA__
        #pragma _NEC vector
        #endif
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            reg_widths[i] = 0.0;
            reg_changes[i] = 0;
        }
        
        int thread_id = omp_get_thread_num();
        _TEdgeWeight *private_widths = &cached_widths[thread_id * CACHED_VERTICES * CACHE_STEP];
        
        while(changes > 0)
        {
            #pragma omp barrier
            
            _reversed_graph.template place_data_into_cache<_TEdgeWeight>(_widths, private_widths);
            
            #pragma omp barrier
            
            #ifdef __USE_NEC_SX_AURORA__
            #pragma _NEC vector
            #endif
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                reg_changes[i] = 0;
            }
            
            #pragma omp master
            {
                iterations_count++;
            }
            changes = 0;
            
            if(number_of_vertices_in_first_part > 0)
            {
                int local_changes = 0;
                for(int src_id = 0; src_id < number_of_vertices_in_first_part; src_id++)
                {
                    long long edge_start = first_part_ptrs[src_id];
                    int connections_count = first_part_sizes[src_id];
                    
                    #ifdef __USE_NEC_SX_AURORA__
                    #pragma _NEC vector
                    #endif
                    for(int i = 0; i < VECTOR_LENGTH; i++)
                    {
                        reg_widths[i] = _widths[src_id];
                    }
                    
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
                            _TEdgeWeight weight = incoming_weights[edge_start + edge_pos + i];
                            _TEdgeWeight width_to_dst = _reversed_graph.template load_vertex_data_cached<_TEdgeWeight>(dst_id, _widths, private_widths);
                            _TEdgeWeight new_width = weight;
                            if(new_width > width_to_dst)
                            new_width = width_to_dst;
                            
                            if(reg_widths[i] < new_width)
                            {
                                reg_widths[i] = new_width;
                            }
                        }
                    }
                    
                    _TEdgeWeight max_width = 0.0;
                    #ifdef __USE_NEC_SX_AURORA__
                    #pragma _NEC vector
                    #endif
                    for(int i = 0; i < VECTOR_LENGTH; i++)
                    {
                        if(reg_widths[i] < max_width)
                        {
                            max_width = reg_widths[i];
                        }
                    }
                    
                    #pragma omp critical
                    {
                        if(_widths[src_id] < max_width)
                        {
                            _widths[src_id] = max_width;
                            local_changes = 1;
                        }
                    }
                }
                
                #pragma omp atomic
                changes += local_changes;
            }
            
            #pragma omp for schedule(static, 1)
            for(int cur_vector_segment = 0; cur_vector_segment < vector_segments_count; cur_vector_segment++)
            {
                int segment_first_vertex = cur_vector_segment * VECTOR_LENGTH + number_of_vertices_in_first_part;
                
                long long segement_edges_start = vector_group_ptrs[cur_vector_segment];
                int segment_connections_count  = vector_group_sizes[cur_vector_segment];
                
                #ifdef __USE_NEC_SX_AURORA__
                #pragma _NEC vector
                #endif
                for(int i = 0; i < VECTOR_LENGTH; i++)
                {
                    int src_id = segment_first_vertex + i;
                    reg_widths[i] = _widths[src_id];
                }
                
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
                        int dst_id = incoming_ids[segement_edges_start + edge_pos * VECTOR_LENGTH + i];
                        _TEdgeWeight weight = incoming_weights[segement_edges_start + edge_pos * VECTOR_LENGTH + i];
                        
                        _TEdgeWeight width_to_dst = _reversed_graph.template load_vertex_data_cached<_TEdgeWeight>(dst_id, _widths, private_widths);
                        _TEdgeWeight new_width = weight;
                        if(new_width > width_to_dst)
                            new_width = width_to_dst;
                        
                        if(reg_widths[i] < new_width)
                        {
                            reg_widths[i] = new_width;
                        }
                    }
                }
                
                #ifdef __USE_NEC_SX_AURORA__
                #pragma _NEC vector
                #endif
                for(int i = 0; i < VECTOR_LENGTH; i++)
                {
                    int src_id = segment_first_vertex + i;
                    if(_widths[src_id] < reg_widths[i])
                    {
                        _widths[src_id] = reg_widths[i];
                        reg_changes[i] = 1;
                    }
                }
            }
            
            #pragma omp barrier
            
            int private_changes = 0;
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                private_changes += reg_changes[i];
            }
            
            #pragma omp atomic
            changes += private_changes;
            
            #pragma omp barrier
        }
    }
    double t2 = omp_get_wtime();
    
    cout << "SSWP time: " << t2 - t1 << endl;
    cout << "SSWP Perf: " << ((double)edges_count) / ((t2 - t1) * 1e6) << " MFLOPS" << endl;
    cout << "SSWP iterations count: " << iterations_count << endl;
    cout << "SSWP Perf per iteration: " << iterations_count * ((double)edges_count) / ((t2 - t1) * 1e6) << " MFLOPS" << endl;
    cout << "SSWP bandwidth: " << ((double)iterations_count)*((double)edges_count * (sizeof(int) + 2*sizeof(_TEdgeWeight))) / ((t2 - t1) * 1e9) << " gb/s" << endl << endl;
    
    _reversed_graph.template free_data<_TEdgeWeight>(cached_widths);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void WidestPaths<_TVertexValue, _TEdgeWeight>::bellman_ford(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_reversed_graph,
                                                            int _source_vertex,
                                                            _TEdgeWeight *_widths)
{
    int vertices_count    = _reversed_graph.get_vertices_count();
    long long edges_count = _reversed_graph.get_edges_count   ();
    long long    *outgoing_ptrs    = _reversed_graph.get_outgoing_ptrs   ();
    int          *outgoing_ids     = _reversed_graph.get_outgoing_ids    ();
    _TEdgeWeight *outgoing_weights = _reversed_graph.get_outgoing_weights();
    int threads_count = omp_get_max_threads();
    
    #ifdef __USE_NEC_SX_AURORA__
    #pragma _NEC retain(_widths)
    #endif
    
    for(int i = 0; i < vertices_count; i++)
    {
        _widths[i] = 0.0;
    }
    _widths[_source_vertex] = FLT_MAX;

    int changes = 1;
    int current_iteration = 1;
    while(changes)
    {
        changes = 0;
        #pragma omp parallel for schedule(static) shared(changes)
        for(int src_id = 0; src_id < vertices_count; src_id++)
        {
            long long edge_start = outgoing_ptrs[src_id];
            int connections_count = outgoing_ptrs[src_id + 1] - outgoing_ptrs[src_id];
            
            _TEdgeWeight old_width = _widths[src_id];
            _TEdgeWeight max_width = old_width;
            
            for(int edge_pos = 0; edge_pos < connections_count; edge_pos++)
            {
                int dst_id = outgoing_ids[edge_start + edge_pos];
                _TEdgeWeight weight = outgoing_weights[edge_start + edge_pos];
                _TEdgeWeight width_to_dst = _widths[dst_id];
                
                if(max_width < min(weight, width_to_dst))
                {
                    max_width = min(weight, width_to_dst);
                }
            }
            
            if(old_width < max_width)
            {
                _widths[src_id] = max_width;
                changes = 1;
            }
        }
        
        current_iteration++;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* bellman_ford_hpp */
