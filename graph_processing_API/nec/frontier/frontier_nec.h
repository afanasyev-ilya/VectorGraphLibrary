#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../../framework_types.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class FrontierNEC // TODO inheritance
{
private:
    // pointer to base graph
    VectCSRGraph *graph_ptr;
    TraversalDirection direction;

    int *flags;
    int *ids;
    int *work_buffer;

    FrontierType type;

    int vector_engine_part_size;
    int vector_core_part_size;
    int collective_part_size;

    FrontierType vector_engine_part_type;
    FrontierType vector_core_part_type;
    FrontierType collective_part_type;

    int current_size;
    int max_size;
public:
    FrontierNEC(VectCSRGraph &_graph, TraversalDirection _direction);
    ~FrontierNEC();

    // get information about frontier API
    int size() {return current_size;};
    FrontierType get_type() {return type;};

    // printing API
    void print_frontier_info();

    // frontier modification API
    inline void set_all_active();
    inline void add_vertex(int src_id);
    inline void add_vertices(int *_vertex_ids, int _number_of_vertices);
    inline void clear() { current_size = 0; };

    // frontier direction API
    TraversalDirection get_direction() {return direction;};
    void set_direction(TraversalDirection _direction) {direction = _direction;};

    friend class GraphAbstractionsNEC;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "frontier_nec.hpp"
#include "modification.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
