//
//  bellman_ford_vectorised_CSR.hpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 24/04/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef bellman_ford_vectorised_CSR_hpp
#define bellman_ford_vectorised_CSR_hpp

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ShortestPaths<_TVertexValue, _TEdgeWeight>::bellman_ford(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight>
                                                              &_reversed_graph,
                                                              int _source_vertex,
                                                              _TEdgeWeight *_distances)
{
    LOAD_VECTORISED_CSR_GRAPH_REVERSE_DATA(_reversed_graph)
    
    int threads_count = omp_get_max_threads();
    _reversed_graph.set_threads_count(threads_count);
    
    #ifdef __USE_NEC_SX_AURORA__
    #pragma _NEC retain(_distances)
    #endif
    
    _reversed_graph.template vertex_array_set_to_constant<_TEdgeWeight>(_distances, FLT_MAX);
    _reversed_graph.template vertex_array_set_element<_TEdgeWeight>(_distances, _source_vertex, 0.0);
    
    _TEdgeWeight *cached_distances = _reversed_graph.template allocate_private_caches<_TEdgeWeight>(threads_count);
    
    double t1 = omp_get_wtime();
    int changes = 1;
    int iterations_count = 0;
    #pragma omp parallel num_threads(threads_count) shared(changes)
    {
        int reg_changes[VECTOR_LENGTH];
        _TEdgeWeight reg_distances[VECTOR_LENGTH];
        
        #ifdef __USE_NEC_SX_AURORA__
        #pragma _NEC vreg(reg_changes)
        #pragma _NEC vreg(reg_distances)
        #endif
        
        #ifdef __USE_NEC_SX_AURORA__
        #pragma _NEC vector
        #endif
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            reg_distances[i] = 0.0;
            reg_changes[i] = 0;
        }
        
        int thread_id = omp_get_thread_num();
        _TEdgeWeight *private_distances = &cached_distances[thread_id * CACHED_VERTICES * CACHE_STEP];
        
        while(changes > 0)
        {
            #pragma omp barrier
            
            _reversed_graph.template place_data_into_cache<_TEdgeWeight>(_distances, private_distances);
            
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
                    _TEdgeWeight shortest_distance = _distances[src_id];
                    long long edge_start = first_part_ptrs[src_id];
                    int connections_count = first_part_sizes[src_id];
                    
                    #ifdef __USE_NEC_SX_AURORA__
                    #pragma _NEC vector
                    #endif
                    for(int i = 0; i < VECTOR_LENGTH; i++)
                    {
                        reg_distances[i] = shortest_distance;
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
                            _TEdgeWeight new_weight = weight + _reversed_graph.template load_vertex_data_cached<_TEdgeWeight>(dst_id, _distances, private_distances);
                            
                            if(new_weight < reg_distances[i])
                                reg_distances[i] = new_weight;
                        }
                    }
                    
                    shortest_distance = FLT_MAX;
                    #ifdef __USE_NEC_SX_AURORA__
                    #pragma _NEC vector
                    #endif
                    for(int i = 0; i < VECTOR_LENGTH; i++)
                    {
                        if(reg_distances[i] < shortest_distance)
                            shortest_distance = reg_distances[i];
                    }
                    
                    #pragma omp critical
                    {
                        if(_distances[src_id] > shortest_distance)
                        {
                            _distances[src_id] = shortest_distance;
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
                    reg_distances[i] = _distances[src_id];
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
                        int src_id = segment_first_vertex + i;
                        int dst_id = incoming_ids[segement_edges_start + edge_pos * VECTOR_LENGTH + i];
                        
                        _TEdgeWeight weight = incoming_weights[segement_edges_start + edge_pos * VECTOR_LENGTH + i];
                        _TEdgeWeight new_weight = weight +_reversed_graph.template load_vertex_data_cached<_TEdgeWeight>(dst_id, _distances, private_distances);
                        
                        if(reg_distances[i] > new_weight)
                        {
                            reg_distances[i] = new_weight;
                            reg_changes[i] = 1;
                        }
                    }
                }
                
                #ifdef __USE_NEC_SX_AURORA__
                #pragma _NEC vector
                #endif
                for(int i = 0; i < VECTOR_LENGTH; i++)
                {
                    int src_id = segment_first_vertex + i;
                    _distances[src_id] = reg_distances[i];
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
    
    cout << "SSSP time: " << t2 - t1 << endl;
    cout << "SSSP Perf: " << ((double)edges_count) / ((t2 - t1) * 1e6) << " MFLOPS" << endl;
    cout << "SSSP iterations count: " << iterations_count << endl;
    cout << "SSSP Perf per iteration: " << iterations_count * ((double)edges_count) / ((t2 - t1) * 1e6) << " MFLOPS" << endl;
    cout << "SSSP bandwidth: " << ((double)iterations_count)*((double)edges_count * (sizeof(int) + 2*sizeof(_TEdgeWeight))) / ((t2 - t1) * 1e9) << " gb/s" << endl << endl;
    
    _reversed_graph.template free_data<_TEdgeWeight>(cached_distances);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _TVertexValue, typename _TEdgeWeight>
void ShortestPaths<_TVertexValue, _TEdgeWeight>::gpu_bellman_ford(
                                                VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_reversed_graph,
                                                int _source_vertex, _TEdgeWeight *_distances)
{
    LOAD_VECTORISED_CSR_GRAPH_REVERSE_DATA(_reversed_graph)
    
    _TEdgeWeight *device_distances;
    SAFE_CALL(cudaMalloc((void**)&device_distances, vertices_count * sizeof(_TEdgeWeight)));
    
    int iterations_count = 0;
    double t1 = omp_get_wtime();
    gpu_bellman_ford_wrapper<_TVertexValue, _TEdgeWeight>(_reversed_graph, device_distances, _source_vertex, iterations_count);
    double t2 = omp_get_wtime();
    
    cout << "GPU time: " << t2 - t1 << endl;
    cout << "GPU Perf: " << ((double)edges_count) / ((t2 - t1) * 1e6) << " MFLOPS" << endl;
    cout << "GPU iterations count: " << iterations_count << endl;
    cout << "GPU Perf per iteration: " << iterations_count * ((double)edges_count) / ((t2 - t1) * 1e6) << " MFLOPS" << endl;
    cout << "GPU bandwidth: " << ((double)iterations_count)*((double)edges_count * (sizeof(int) + 2*sizeof(_TEdgeWeight))) / ((t2 - t1) * 1e9) << " gb/s" << endl << endl;
    
    SAFE_CALL(cudaMemcpy(_distances, device_distances, vertices_count * sizeof(_TEdgeWeight), cudaMemcpyDeviceToHost));
    
    SAFE_CALL(cudaFree(device_distances));
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* bellman_ford_vectorised_CSR_hpp */
