//
//  page_rank.h
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 24/04/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef page_rank_h
#define page_rank_h

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
class PageRank
{
private:
    static double calculate_ranks_sum(float *_page_ranks, int _vertices_count);
    static bool algorithm_converged(float *_current_page_ranks, float *_old_page_ranks, int _vertices_count,
                                    float _convergence_factor);
    
    static void revert_outgoing_size_values(float *_result, int *_input, int _vertices_count, int _threads_count);
    
    static void page_rank_kernel(long long *_first_part_ptrs, int *_first_part_sizes,
                                 int _number_of_vertices_in_first_part, long long *_vector_group_ptrs,
                                 int *_vector_group_sizes, int *_incoming_ids, float *_page_ranks,
                                 float *_gather_buffer, float *_reversed_outgoing_size_per_edge,
                                 int _vector_segments_count, float _k, float _d,  int _threads_count,
                                 float _dangling_input);
    
    static void calculate_loops_number(int *_number_of_loops, long long *_first_part_ptrs, int *_first_part_sizes,
                                       int _number_of_vertices_in_first_part, long long int *_vector_group_ptrs,
                                       int *_vector_group_sizes, int *_incoming_ids, int _vector_segments_count,
                                       int _threads_count);
    
    static void get_outgoing_sizes_without_loops(int *_number_of_loops, int *_outgoing_sizes, int _vertices_count,
                                                 int _threads_count);
    
    static void print_performance_stats(long long _edges_count, int _iterations_count, float _total_time,
                                        float _gather_time, int _bytes_per_edge);
    
    static float calculate_dangling_input(int *_outgoing_sizes, float *_old_page_ranks, int _vertices_count,
                                           int _threads_count);
public:
    static void allocate_result_memory  (int _vertices_count, float **_page_ranks);
    static void free_result_memory      (float *_page_ranks);

    static void page_rank_cached(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_reversed_graph, float *_page_ranks,
                          int _threads_count, float _convergence_factor = 1.0e-4, int _max_iterations = 100);
    
    static void page_rank(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_reversed_graph, float *_page_ranks,
                          int _threads_count, float _convergence_factor = 1.0e-4, int _max_iterations = 100) {};
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "page_rank.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* page_rank_h */
