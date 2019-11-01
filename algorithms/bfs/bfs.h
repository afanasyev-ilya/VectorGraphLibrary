//
//  bfs.h
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 03/06/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef bfs_h
#define bfs_h

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
#include "gpu/traditional_bfs.h"
#endif

#include <string>
#include "vertex_queue.h"
#include "change_state.h"

#define BOTTOM_UP_THRESHOLD 4
#define BOTTOM_UP_REMINDER_VERTEX -3

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
class BFS
{
private:
    static int number_of_active(int *_levels, int _vertices_count, int _current_level);
    
    static inline int nec_get_active_count(int *_levels, int _vertices_count, int _desired_level);
    static inline void nec_generate_frontier(int *_levels, int *_active_ids, int _vertices_count, int _desired_level,
                                             int _threads_count);
    
    static inline void nec_calculate_balance(int *_levels, int *_active_ids, int _vertices_count, int _desired_level, int _threads_count);

    static inline void nec_top_down_step(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, long long *_outgoing_ptrs,
                                         int *_outgoing_ids, float *_outgoing_weights, int _vertices_count, int _active_count,
                                         int *_levels, int *_cached_levels, int *_active_ids, int _cur_level, int &_vis,
                                         int &_in_lvl, int _threads_count);
    
    static inline void nec_bottom_up_step(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, long long *_outgoing_ptrs,
                                          int *_outgoing_ids, int _vertices_count, int _active_count, int *_levels,
                                          int *_cached_levels, int *_active_ids, int _cur_level, int &_vis, int &_in_lvl,
                                          int _threads_count, int *_partial_outgoing_ids, bool _use_vect_CSR_extension,
                                          int _non_zero_vertices_count, double &_t_first, double &_t_second, double &_t_third);
public:
    static void allocate_result_memory(int _vertices_count, int **_levels);
    static void free_result_memory    (int *_levels);
    
    static double nec_direction_optimising_BFS(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                               int *_levels, int _source_vertex);
    
    #ifdef __USE_GPU__
    static void gpu_direction_optimising_BFS(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                             int *_levels,
                                             int _source_vertex);
    #endif
    
    static void verifier(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int _source_vertex, int *_parallel_levels);
    
    static void verifier(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int _source_vertex, int *_parallel_levels);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "bfs.hpp"
#include "nec_bfs.hpp"
#include "verifier.hpp"
#include "change_state.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* bfs_h */
