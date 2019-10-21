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

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
class BFS
{
private:
    static int number_of_active(int *_levels, int _vertices_count, int _current_level);

    static void nec_top_down_step(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int *_outgoing_ids,
                                  int _number_of_vertices_in_first_part, int *_levels, VertexQueue &_global_queue,
                                  VertexQueue **_local_queues, int _omp_threads, int _current_level, int &_vis,
                                  int &_in_lvl);
    
    static void nec_bottom_up_step(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, long long *_first_part_ptrs,
                                   int *_first_part_sizes, int _vector_segments_count, long long *_vector_group_ptrs,
                                   int *_vector_group_sizes, int *_outgoing_ids, int _vertices_count,
                                   int _number_of_vertices_in_first_part, int *_levels, int _omp_threads, int _current_level, int &_vis,
                                   int &_in_lvl);
    
    static void intel_top_down_step(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int *_outgoing_ids,
                                    int _number_of_vertices_in_first_part, int *_levels, VertexQueue &_global_queue,
                                    VertexQueue **_local_queues, int _omp_threads, int _current_level, int &_vis,
                                    int &_in_lvl);
    
    static void intel_bottom_up_step(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, long long *_first_part_ptrs,
                                     int *_first_part_sizes, int _vector_segments_count, long long *_vector_group_ptrs,
                                     int *_vector_group_sizes, int *_outgoing_ids, int _vertices_count,
                                     int _number_of_vertices_in_first_part, int *_levels, VertexQueue &_global_queue,
                                     VertexQueue **_local_queues, int _omp_threads, int _current_level, int &_vis,
                                     int &_in_lvl);
public:
    static void allocate_result_memory(int _vertices_count, int **_levels);
    static void free_result_memory    (int *_levels);
    
    static void intel_direction_optimising_BFS(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                               int *_levels,
                                               int _source_vertex);
    
    static void nec_direction_optimising_BFS(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                             int *_levels,
                                             int _source_vertex);
    
    static void test_primitives(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                int *_levels,
                                int _source_vertex);
    
    static void new_bfs(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int *_levels, int _source_vertex);
    
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
#include "verifier.hpp"
#include "change_state.hpp"
#include "intel/direction_optimising_bfs.hpp"
#include "nec/direction_optimising_bfs.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* bfs_h */
