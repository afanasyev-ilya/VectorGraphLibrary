#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Frontier
{
protected:
    ObjectType object_type;

    // pointer to base graph
    VectCSRGraph *graph_ptr;
    TraversalDirection direction;

    // frontier type - sparse, dense, all-active
    FrontierType type;

    // number of vertices located in frontier
    int current_size;
    int max_size; // TODO remove? can be obtained from graph prt
    long long neighbours_count;
public:
    /* constructors and destructors */
    Frontier();

    // get API
    int size() {return current_size;};
    FrontierType get_type() {return type;};
    ObjectType get_object_type() {return object_type;};
    long long get_neighbours_count() {return neighbours_count;};

    // printing API
    virtual void print_stats() = 0;
    virtual void print() = 0;

    // frontier modification API
    virtual inline void set_all_active() = 0;
    virtual inline void add_vertex(int src_id) = 0;
    virtual inline void add_group_of_vertices(int *_vertex_ids, int _number_of_vertices) = 0;
    inline void clear() { current_size = 0; neighbours_count = 0; };

    // frontier direction API
    TraversalDirection get_direction() {return direction;};
    void set_direction(TraversalDirection _direction) {direction = _direction;};
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "frontier.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
#include "../../nec/frontier/frontier_nec.h"
#endif

#if defined(__USE_GPU__)
#include "../../gpu/frontier/frontier_gpu.cuh"
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
