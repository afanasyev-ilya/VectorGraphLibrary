#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../../framework_types.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class FrontierNEC // TODO inheritance
{
private:
    TraversalDirection frontier_direction;

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
    FrontierNEC(VectCSRGraph &_graph, TraversalDirection _frontier_direction);
    ~FrontierNEC();

    int size() {return current_size;};
    FrontierType get_type() {return type;};

    void set_all_active();

    void change_size(int _size) {max_size = _size;};

    void print_frontier_info(ExtendedCSRGraph &_graph);

    inline void add_vertex(ExtendedCSRGraph &_graph, int src_id);

    inline void add_vertices(ExtendedCSRGraph &_graph, int *_vertex_ids, int _number_of_vertices);

    inline void clear() { current_size = 0; };

    TraversalDirection get_direction() {return frontier_direction;};
    void set_direction(TraversalDirection _new_direction) {frontier_direction = _new_direction;};

    friend class GraphAbstractionsNEC;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "frontier_nec.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
