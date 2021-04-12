#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define RW RandomWalk
#define DEAD_END -1

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class RandomWalk
{
public:
    // architecture-independent API example. same function for NEC, multicore, GPU
    template <typename _T>
    static void vgl_random_walk(VectCSRGraph &_graph,
                                vector<int> &_walk_vertices,
                                int _walk_vertices_num,
                                int _walk_lengths,
                                VerticesArray<_T> &_walk_results);

    template <typename _T>
    static void seq_random_walk(VectCSRGraph &_graph,
                                vector<int> &_walk_vertices,
                                int _walk_vertices_num,
                                int _walk_lengths,
                                VerticesArray<_T> &_walk_results);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "random_walk.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
