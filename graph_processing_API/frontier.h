#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "framework_types.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Frontier
{
private:
    int *flags;
    int *ids;

    FrontierType type;

    int current_size;
    int max_size;
public:
    int size() {return current_size;};
    FrontierType get_type() {return type;};

    void set_all_active() {};

    void change_size(int _size) {max_size = _size;};


    void print_frontier_info(ExtendedCSRGraph &_graph) {};


    inline void add_vertex(ExtendedCSRGraph &_graph, int src_id) {};


    inline void add_vertices(ExtendedCSRGraph &_graph,
                                     int *_vertex_ids,
                                     int _number_of_vertices) {};

    inline void clear() { current_size = 0; };
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "nec/graph_primitives/graph_primitives_nec.h"
#include "multicore/graph_primitives/graph_primitives_multicore.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
