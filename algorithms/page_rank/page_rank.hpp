//
//  page_rank.hpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 24/04/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef page_rank_hpp
#define page_rank_hpp

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void PageRank<_TVertexValue, _TEdgeWeight>::allocate_result_memory(int _vertices_count, float **_page_ranks)
{
    *_page_ranks = new float[_vertices_count];
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void PageRank<_TVertexValue, _TEdgeWeight>::free_result_memory(float *_page_ranks)
{
    delete[] _page_ranks;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
double PageRank<_TVertexValue, _TEdgeWeight>::calculate_ranks_sum(float *_page_ranks, int _vertices_count)
{
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for(int i = 0; i < _vertices_count; i++)
    {
        sum += (double)_page_ranks[i];
    }
    return sum;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
bool PageRank<_TVertexValue, _TEdgeWeight>::algorithm_converged(float *_current_page_ranks, float *_old_page_ranks,
                                                                int _vertices_count, float _convergence_factor)
{
    float err = 0.0;
    #pragma omp parallel for reduction(+:err)
    for(int i = 0; i < _vertices_count; i++)
    {
        float diff = _current_page_ranks[i] - _old_page_ranks[i];
        if(diff < 0)
            diff *= -1;
        err += diff;
    }
    
    if(err < (_vertices_count * _convergence_factor))
        return true;
    
    return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void PageRank<_TVertexValue, _TEdgeWeight>::revert_outgoing_size_values(float *_result, int *_input, int _vertices_count,
                                                                        int _threads_count)
{
    #pragma omp parallel for num_threads(_threads_count)
    for(int i = 0; i < _vertices_count; i++)
        _result[i] = 1.0 / ((float)_input[i]);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void PageRank<_TVertexValue, _TEdgeWeight>::page_rank_kernel(long long *_first_part_ptrs,
                                                             int *_first_part_sizes,
                                                             int _number_of_vertices_in_first_part,
                                                             long long *_vector_group_ptrs,
                                                             int *_vector_group_sizes,
                                                             int *_incoming_ids,
                                                             float *_page_ranks,
                                                             float *_gather_buffer,
                                                             float *_reversed_outgoing_size_per_edge,
                                                             int _vector_segments_count,
                                                             float _k,
                                                             float _d,
                                                             int _threads_count,
                                                             float _dangling_input)
{
    float rank_sum = 0.0;
    #pragma omp parallel num_threads(_threads_count) shared(rank_sum)
    {
        float reg_ranks[VECTOR_LENGTH];
        
        #ifdef __USE_NEC_SX_AURORA__
        #pragma vreg(reg_ranks)
        #endif
        
        #ifdef __USE_NEC_SX_AURORA__
        #pragma _NEC vector
        #endif
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            reg_ranks[i] = 0;
        }
        
        #pragma omp for schedule(static, 1)
        for(int cur_vector_segment = 0; cur_vector_segment < _vector_segments_count; cur_vector_segment++)
        {
            int segment_first_vertex = cur_vector_segment * VECTOR_LENGTH + _number_of_vertices_in_first_part;
            
            long long segement_edges_start = _vector_group_ptrs[cur_vector_segment];
            int segment_connections_count  = _vector_group_sizes[cur_vector_segment];
            
            #ifdef __USE_NEC_SX_AURORA__
            #pragma _NEC vector
            #endif
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                reg_ranks[i] = 0;
            }
            
            for(int edge_pos = 0; edge_pos < segment_connections_count; edge_pos++)
            {
                #ifdef __USE_NEC_SX_AURORA__
                #pragma _NEC ivdep
                #pragma unroll
                #pragma vector
                #endif
                for(int i = 0; i < VECTOR_LENGTH; i++)
                {
                    int src_id = segment_first_vertex + i;
                    int dst_id = _incoming_ids[segement_edges_start + edge_pos * VECTOR_LENGTH + i];
                    
                    float dst_rank = _gather_buffer[segement_edges_start + edge_pos * VECTOR_LENGTH + i];
                    float dst_links_num = _reversed_outgoing_size_per_edge[segement_edges_start + edge_pos * VECTOR_LENGTH + i];
                    
                    if(src_id != dst_id)
                        reg_ranks[i] += dst_rank * dst_links_num;
                }
            }
            
            #ifdef __USE_NEC_SX_AURORA__
            #pragma _NEC ivdep
            #pragma unroll
            #pragma vector
            #endif
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                int src_id = segment_first_vertex + i;
                _page_ranks[src_id] = _k + _d * (reg_ranks[i] + _dangling_input);
            }
        }
        
        for(int src_id = 0; src_id < _number_of_vertices_in_first_part; src_id++)
        {
            #pragma omp barrier
            
            long long edge_start = _first_part_ptrs[src_id];
            int connections_count = _first_part_sizes[src_id];
            
            #ifdef __USE_NEC_SX_AURORA__
            #pragma _NEC vector
            #endif
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                reg_ranks[i] = 0.0;
            }
            
            rank_sum = 0.0;
            
            #pragma omp for schedule(static, 1) reduction(+:rank_sum)
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
                    int dst_id = _incoming_ids[edge_start + edge_pos + i];
                    float dst_rank = _gather_buffer[edge_start + edge_pos + i];
                    float dst_links_num = _reversed_outgoing_size_per_edge[edge_start + edge_pos + i];
                    
                    if(src_id != dst_id)
                        rank_sum += dst_rank * dst_links_num;
                }
            }
            
            /*#pragma omp critical
            {
                #pragma _NEC vector
                for(int i = 0; i < VECTOR_LENGTH; i++)
                {
                    rank_sum += reg_ranks[i];
                }
            }*/
            
            #pragma omp master
            {
                _page_ranks[src_id] = _k + _d * (rank_sum + _dangling_input);
            }
            
            #pragma omp barrier
        }
    }
    
    /*#pragma omp parallel
    {
        #pragma omp for schedule(static, 1)
        for(int src_id = 0; src_id < _number_of_vertices_in_first_part; src_id++)
        {
            long long edge_start = _first_part_ptrs[src_id];
            int connections_count = _first_part_sizes[src_id];
            
            float reg_rank = 0;
            
            for(int edge_pos = 0; edge_pos < connections_count; edge_pos++)
            {
                int dst_id = _incoming_ids[edge_start + edge_pos];
                float dst_rank = _gather_buffer[edge_start + edge_pos];
                float dst_links_num = _reversed_outgoing_size_per_edge[edge_start + edge_pos];
                
                if(src_id != dst_id)
                    reg_rank += dst_rank * dst_links_num;
            }
            
            _page_ranks[src_id] = _k + _d * (reg_rank + _dangling_input);
        }
    }*/
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void PageRank<_TVertexValue, _TEdgeWeight>::calculate_loops_number(int *_number_of_loops,
                                                                   long long *_first_part_ptrs,
                                                                   int *_first_part_sizes,
                                                                   int _number_of_vertices_in_first_part,
                                                                   long long int *_vector_group_ptrs,
                                                                   int *_vector_group_sizes,
                                                                   int *_incoming_ids,
                                                                   int _vector_segments_count,
                                                                   int _threads_count)
{
    // process last part first
    #pragma omp parallel num_threads(_threads_count)
    {
        int reg_loops[VECTOR_LENGTH];
        
        #pragma omp for schedule(static, 1)
        for(int cur_vector_segment = 0; cur_vector_segment < _vector_segments_count; cur_vector_segment++)
        {
            int segment_first_vertex = cur_vector_segment * VECTOR_LENGTH + _number_of_vertices_in_first_part;
            long long segement_edges_start = _vector_group_ptrs[cur_vector_segment];
            int segment_connections_count  = _vector_group_sizes[cur_vector_segment];
            
            #ifdef __USE_NEC_SX_AURORA__
            #pragma _NEC ivdep
            #pragma unroll
            #endif
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                reg_loops[i] = 0;
            }
            
            for(int edge_pos = 0; edge_pos < segment_connections_count; edge_pos++)
            {
                #ifdef __USE_NEC_SX_AURORA__
                #pragma _NEC ivdep
                #pragma unroll
                #pragma vector
                #endif
                for(int i = 0; i < VECTOR_LENGTH; i++)
                {
                    int src_id = segment_first_vertex + i;
                    int dst_id = _incoming_ids[segement_edges_start + edge_pos * VECTOR_LENGTH + i];
                    
                    if(src_id == dst_id)
                        reg_loops[i]++;
                }
            }
            
            #ifdef __USE_NEC_SX_AURORA__
            #pragma _NEC ivdep
            #pragma unroll
            #endif
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                int src_id = segment_first_vertex + i;
                _number_of_loops[src_id] = reg_loops[i];
            }
        }
    }

    // process first part after
    #pragma omp parallel for schedule(static)
    for(int src_id = 0; src_id < _number_of_vertices_in_first_part; src_id++)
    {
        long long edge_start = _first_part_ptrs[src_id];
        int connections_count = _first_part_sizes[src_id];
        
        int reg_loops = 0;
        for(int edge_pos = 0; edge_pos < connections_count; edge_pos++)
        {
            int dst_id = _incoming_ids[edge_start + edge_pos];
            
            if(src_id == dst_id)
                reg_loops++;
        }
        
        _number_of_loops[src_id] = reg_loops;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void PageRank<_TVertexValue, _TEdgeWeight>::get_outgoing_sizes_without_loops(int *_number_of_loops,
                                                                             int *_outgoing_sizes,
                                                                             int _vertices_count,
                                                                             int _threads_count)
{
    #pragma omp parallel for num_threads(_threads_count)
    for(int i = 0; i < _vertices_count; i++)
    {
        _outgoing_sizes[i] -= _number_of_loops[i];
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
float PageRank<_TVertexValue, _TEdgeWeight>::calculate_dangling_input(int *_outgoing_sizes, float *_old_page_ranks,
                                                                       int _vertices_count, int _threads_count)
{
    float dangling_input = 0;
    #pragma omp parallel for reduction(+: dangling_input) num_threads(_threads_count)
    for(int i = 0; i < _vertices_count; i++)
    {
        if(_outgoing_sizes[i] <= 0)
        {
            dangling_input += _old_page_ranks[i] / ((float)_vertices_count);
        }
    }
    return dangling_input;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void PageRank<_TVertexValue, _TEdgeWeight>::print_performance_stats(long long _edges_count,
                                                                    int _iterations_count,
                                                                    float _total_time,
                                                                    float _gather_time,
                                                                    int _bytes_per_edge)
{
    cout << "PR wall time: " << _total_time << endl;
    cout << "PR wall Perf: " << ((float)_edges_count) / (_total_time * 1e6) << " MFLOPS" << endl;
    //cout << "PR gather perf: " << _iterations_count * ((float)_edges_count) / (_gather_time * 1e6) << " MFLOPS" << endl;
    cout << "PR iterations count: " << _iterations_count << endl;
    cout << "PR per iteration: " << _iterations_count * ((float)_edges_count) / (_total_time * 1e6) << " MFLOPS" << endl << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void PageRank<_TVertexValue, _TEdgeWeight>::page_rank_cached(
                                                      VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_reversed_graph,
                                                      float *_page_ranks,
                                                      int _threads_count,
                                                      float _convergence_factor,
                                                      int _max_iterations)
{
    LOAD_VECTORISED_CSR_GRAPH_REVERSE_DATA(_reversed_graph)
    
    int    *number_of_loops                    = _reversed_graph.template vertex_array_alloc<int>   ();
    float *reversed_outgoing_sizes_per_vertex = _reversed_graph.template vertex_array_alloc<float>();
    float *old_page_ranks                     = _reversed_graph.template vertex_array_alloc<float>();
    float *reversed_outgoing_sizes_per_edge   = _reversed_graph.template edges_array_alloc <float>();
    float *gather_buffer                      = _reversed_graph.template edges_array_alloc <float>();
    
    float *cached_data = _reversed_graph.template allocate_private_caches<float>(_threads_count);
    
    float d = 0.85;
    float k = (1.0 - d) / ((float)vertices_count);
    double t1 = 0, t2 = 0, total_time = 0, gather_time = 0;
    
    _reversed_graph.set_threads_count(_threads_count);
    _reversed_graph.template vertex_array_set_to_constant<int>(number_of_loops, 0.0);
    
    // init and prepare datastructures
    calculate_loops_number(number_of_loops, first_part_ptrs, first_part_sizes, number_of_vertices_in_first_part,
                           vector_group_ptrs, vector_group_sizes, incoming_ids, vector_segments_count,
                           _threads_count);
    t1 = omp_get_wtime();
    get_outgoing_sizes_without_loops(number_of_loops, outgoing_sizes, vertices_count, _threads_count);
    _reversed_graph.template free_data<int>(number_of_loops);
    revert_outgoing_size_values(reversed_outgoing_sizes_per_vertex, outgoing_sizes, vertices_count, _threads_count);
    _reversed_graph.template gather_all_edges_data_cached<float>(reversed_outgoing_sizes_per_edge,
                                                                  reversed_outgoing_sizes_per_vertex,
                                                                  cached_data);
    _reversed_graph.template free_data<float>(reversed_outgoing_sizes_per_vertex);
    t2 = omp_get_wtime();
    total_time += t2 - t1;
    cout << "Prepare outgoings per edge time: " << (t2 - t1) * 1000.0 << endl;
    
    // launch algorithm
    double dang_time = 0, kernel_time = 0, converge_time = 0;
    double t3 = omp_get_wtime();
    _reversed_graph.template vertex_array_set_to_constant<float>(_page_ranks, 1.0/((float)vertices_count));
    int iterations_count = 0;
    for(iterations_count = 0; iterations_count < _max_iterations; iterations_count++)
    {
        _reversed_graph.template vertex_array_copy<float>(old_page_ranks, _page_ranks);
        
        double t5 = omp_get_wtime();
        _reversed_graph.template gather_all_edges_data_cached<float>(gather_buffer, old_page_ranks, cached_data);
        double t6 = omp_get_wtime();
        gather_time += t6 - t5;
        
        t5 = omp_get_wtime();
        float dangling_input = calculate_dangling_input(outgoing_sizes, old_page_ranks, vertices_count, _threads_count);
        t6 = omp_get_wtime();
        dang_time += t6 - t5;
        
        t5 = omp_get_wtime();
        page_rank_kernel(first_part_ptrs, first_part_sizes, number_of_vertices_in_first_part,
                         vector_group_ptrs, vector_group_sizes, incoming_ids, _page_ranks,
                         gather_buffer, reversed_outgoing_sizes_per_edge, vector_segments_count, k, d,
                         _threads_count, dangling_input);
        t6 = omp_get_wtime();
        kernel_time += t6 - t5;
    
        double ranks_sum = calculate_ranks_sum(_page_ranks, vertices_count);
        if(fabs(ranks_sum - 1.0) > _convergence_factor)
        {
            cout << "ranks sum: " << ranks_sum << endl;
            throw "ERROR: page rank sum is incorrect";
        }
        
        if(algorithm_converged(_page_ranks, old_page_ranks, vertices_count, _convergence_factor))
        {
            iterations_count++;
            break;
        }
    }
    double t4 = omp_get_wtime();
    total_time += t4 - t3;
    
    //cout << "PR gather perf: " << iterations_count * ((float)edges_count) / (gather_time * 1e6) << " MFLOPS" << endl;
    //cout << "PR kernel perf: " << iterations_count * ((float)edges_count) / (kernel_time * 1e6) << " MFLOPS" << endl << endl;
    
    print_performance_stats(edges_count, iterations_count, total_time, gather_time, 4 * sizeof(float) + sizeof(int));
    
    _reversed_graph.template free_data<float>(reversed_outgoing_sizes_per_edge);
    _reversed_graph.template free_data<float>(old_page_ranks);
    _reversed_graph.template free_data<float>(cached_data);
    _reversed_graph.template free_data<float>(gather_buffer);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#endif /* page_rank_hpp */
