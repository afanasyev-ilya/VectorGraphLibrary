#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class FrontierNEC : public Frontier
{
private:
    // this is how NEC frontier is represented
    int *flags;
    int *ids;
    int *work_buffer;

    int vector_engine_part_size;
    int vector_core_part_size;
    int collective_part_size;

    FrontierType vector_engine_part_type;
    FrontierType vector_core_part_type;
    FrontierType collective_part_type;
public:
    FrontierNEC(VectCSRGraph &_graph, TraversalDirection _direction);
    ~FrontierNEC();

    // printing API
    void print_stats();
    void print();

    // frontier modification API
    inline void set_all_active();
    inline void add_vertex(int src_id);
    inline void add_group_of_vertices(int *_vertex_ids, int _number_of_vertices);

    friend class GraphAbstractionsNEC;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "frontier_nec.hpp"
#include "modification.hpp"
#include "print.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
