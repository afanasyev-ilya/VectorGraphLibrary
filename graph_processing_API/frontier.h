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
public:
    /* constructors and destructors */
    Frontier() {object_type = FRONTIER;};
    ~Frontier() {};

    // get API
    int size() {return current_size;};
    FrontierType get_type() {return type;};
    ObjectType get_object_type() {return object_type;};

    // printing API
    virtual void print_stats() = 0;
    virtual void print() = 0;

    // frontier modification API
    virtual inline void set_all_active() = 0;
    virtual inline void add_vertex(int src_id) = 0;
    virtual inline void add_group_of_vertices(int *_vertex_ids, int _number_of_vertices) = 0;
    inline void clear() { current_size = 0; };

    // frontier direction API
    TraversalDirection get_direction() {return direction;};
    void set_direction(TraversalDirection _direction) {direction = _direction;};
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

string get_frontier_status_string(FrontierType _type)
{
    string status;
    if(_type == ALL_ACTIVE_FRONTIER)
        status = "all active";
    if(_type == SPARSE_FRONTIER)
        status = "sparse";
    if(_type == DENSE_FRONTIER)
        status = "dense";

    return status;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "nec/frontier/frontier_nec.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
