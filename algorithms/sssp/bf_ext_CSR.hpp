//
//  bf_ext_CSR.hpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 21/08/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef bf_ext_CSR_hpp
#define bf_ext_CSR_hpp

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ShortestPaths<_TVertexValue, _TEdgeWeight>::bellman_ford(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                              int _source_vertex,
                                                              _TEdgeWeight *_distances) 
{
    int vertices_count    = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count   ();
    
    long long    *outgoing_ptrs    = _graph.get_outgoing_ptrs   ();
    int          *outgoing_ids     = _graph.get_outgoing_ids    ();
    _TEdgeWeight *outgoing_weights = _graph.get_outgoing_weights();
    
    const int threads_count = omp_get_max_threads();
    #pragma omp parallel for num_threads(threads_count)
    for(int i = 0; i < vertices_count; i++)
    {
        _distances[i] = FLT_MAX;
    }
    _distances[_source_vertex] = 0;
    
    double t1 = omp_get_wtime();
    int changes = 1;
    int iterations_count = 0;
    #pragma omp parallel num_threads(threads_count) shared(changes)
    {
        long long reg_edge_pos[VECTOR_LENGTH];
        int reg_connections_count[VECTOR_LENGTH];
        int reg_changes[VECTOR_LENGTH];
        _TEdgeWeight reg_distances[VECTOR_LENGTH];
        
        #ifdef __USE_NEC_SX_AURORA__
        #pragma _NEC vector
        #pragma vreg(reg_edge_pos)
        #pragma vreg(reg_connections_count)
        #pragma vreg(reg_changes)
        #pragma vreg(reg_distances)
        #endif
        
        while(changes)
        {
            #pragma omp barrier
            
            changes = 0;
            
            #ifdef __USE_NEC_SX_AURORA__
            #pragma _NEC vector
            #endif
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                reg_changes[i] = 0;
            }
            
            #pragma omp for schedule(static)
            for(int starting_vertex = 0; starting_vertex < vertices_count; starting_vertex += VECTOR_LENGTH)
            {
                #ifdef __USE_NEC_SX_AURORA__
                #pragma _NEC vector
                #endif
                for(int i = 0; i < VECTOR_LENGTH; i++)
                {
                    int src_id = starting_vertex + i;
                    reg_distances[i] = _distances[src_id];
                    reg_connections_count[i] = outgoing_ptrs[src_id + 1] - outgoing_ptrs[src_id];
                    reg_edge_pos[i] = outgoing_ptrs[src_id];
                }
                
                int max_connections_count = 0;
                #ifdef __USE_NEC_SX_AURORA__
                #pragma _NEC vector
                #endif
                for(int i = 0; i < VECTOR_LENGTH; i++)
                {
                    if(reg_connections_count[i] > max_connections_count)
                        max_connections_count = reg_connections_count[i];
                }
                
                for(int edge_pos = 0; edge_pos < max_connections_count; edge_pos++)
                {
                    #ifdef __USE_NEC_SX_AURORA__
                    #pragma _NEC vector
                    #pragma simd
                    #pragma ivdep
                    #pragma vovertake
                    #pragma novob
                    #pragma unroll
                    #pragma vector
                    #endif
                    for(int i = 0; i < VECTOR_LENGTH; i++)
                    {
                        if(edge_pos < reg_connections_count[i])
                        {
                            int dst_id = outgoing_ids[reg_edge_pos[i]];
                            _TEdgeWeight weight = outgoing_weights[reg_edge_pos[i]];
                            _TEdgeWeight dst_weight = weight + _distances[dst_id];
                            
                            if(reg_distances[i] > dst_weight)
                            {
                                reg_distances[i] = dst_weight;
                                reg_changes[i] = 1;
                            }
                            
                            reg_edge_pos[i]++;
                        }
                    }
                }
                
                #ifdef __USE_NEC_SX_AURORA__
                #pragma _NEC vector
                #endif
                for(int i = 0; i < VECTOR_LENGTH; i++)
                {
                    int src_id = starting_vertex + i;
                    _distances[src_id] = reg_distances[i];
                }
            }
            
            int private_changes = 0;
            #ifdef __USE_NEC_SX_AURORA__
            #pragma _NEC vector
            #endif
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                private_changes += reg_changes[i];
            }
            
            #pragma omp barrier
            
            #pragma omp atomic
            changes += private_changes;
            
            #pragma omp master
            {
                iterations_count++;
            }
            
            #pragma omp barrier
        }
    }
    double t2 = omp_get_wtime();
    
    #ifdef __PRINT_DETAILED_STATS__
    print_performance_stats(edges_count, iterations_count, t2 - t1);
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* bf_ext_CSR_hpp */
