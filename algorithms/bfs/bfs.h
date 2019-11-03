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

#define BOTTOM_UP_THRESHOLD 3
#define BOTTOM_UP_REMINDER_VERTEX -3

#define UNVISITED_VERTEX -1
#define ISOLATED_VERTEX -2

#define PRINT_DETAILED_STATS

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
class BFS
{
private:
    // temporary buffers
    int *active_ids;
    int *active_vertices_buffer;
    int *vectorised_outgoing_ids;
    
    // nec functions
    int nec_remove_zero_nodes(long long *_outgoing_ptrs, int _vertices_count, int *_levels);
    int nec_mark_zero_nodes(int _vertices_count, int *_levels);
    
    int nec_get_active_count(int *_levels, int _vertices_count, int _desired_level);
    
    int nec_sparse_generate_frontier(int *_levels, int *_active_ids, int _vertices_count, int _desired_level, int _threads_count);

    void nec_top_down_step(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, long long *_outgoing_ptrs,
                           int *_outgoing_ids, float *_outgoing_weights, int _vertices_count, int _active_count,
                           int *_levels, int *_cached_levels, int *_active_ids, int _cur_level, int &_vis,
                           int &_in_lvl, int _threads_count);
    
    void nec_bottom_up_step(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, long long *_outgoing_ptrs,
                            int *_outgoing_ids, int _vertices_count, int _active_count, int *_levels,
                            int *_cached_levels, int *_active_ids, int _cur_level, int &_vis, int &_in_lvl,
                            int _threads_count, bool _use_vect_CSR_extension, int _non_zero_vertices_count,
                            double &_t_first, double &_t_second, double &_t_third);
public:
    BFS();
    ~BFS();
    
    void init_temporary_datastructures(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph);
    
    void allocate_result_memory(int _vertices_count, int **_levels);
    void free_result_memory    (int *_levels);
    
    void nec_direction_optimising_BFS(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int *_levels, int _source_vertex);
    
    #ifdef __USE_GPU__
    void gpu_direction_optimising_BFS(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int *_levels, int _source_vertex);
    #endif
    
    void verifier(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int _source_vertex, int *_parallel_levels);
    
    void verifier(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int _source_vertex, int *_parallel_levels);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "bfs.hpp"
#include "nec_bfs.hpp"
#include "verifier.hpp"
#include "change_state.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* bfs_h */
