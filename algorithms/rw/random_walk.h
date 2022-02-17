#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define RW RandomWalk
#define DEAD_END -1

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// implements random walk algorithm
class RandomWalk
{
public:
    // architecture-independent API example. same function for NEC, multicore, GPU
    #if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
    template <typename _T>
    static void vgl_random_walk(VGL_Graph &_graph,
                                vector<int> &_walk_vertices,
                                int _walk_vertices_num,
                                int _walk_lengths,
                                VerticesArray<_T> &_walk_results);
    #endif

    template <typename _T>
    static void seq_random_walk(VGL_Graph &_graph,
                                vector<int> &_walk_vertices,
                                int _walk_vertices_num,
                                int _walk_lengths,
                                VerticesArray<_T> &_walk_results);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "random_walk.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
