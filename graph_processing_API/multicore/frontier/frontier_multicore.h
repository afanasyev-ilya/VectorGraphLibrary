#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class FrontierMulticore : public Frontier
{
private:
    // this is how NEC frontier is represented
    int *flags;
    int *ids;
    int *work_buffer;

    int vector_engine_part_size;
    int vector_core_part_size;
    int collective_part_size;

    long long vector_engine_part_neighbours_count;
    long long vector_core_part_neighbours_count;
    long long collective_part_neighbours_count;

    FrontierType vector_engine_part_type;
    FrontierType vector_core_part_type;
    FrontierType collective_part_type;

    void init();
public:
    /* constructors and destructors */
    FrontierMulticore(VectCSRGraph &_graph, TraversalDirection _direction = SCATTER);
    FrontierMulticore(ShardedCSRGraph &_graph, TraversalDirection _direction = SCATTER);
    ~FrontierMulticore();

    /* Get API */
    int *get_flags() {return flags;};
    int *get_ids() {return ids;};
    int get_vector_engine_part_size(){return vector_engine_part_size;};
    int get_vector_core_part_size(){return vector_core_part_size;};
    int get_collective_part_size(){return collective_part_size;};

    long long get_vector_engine_part_neighbours_count(){return vector_engine_part_neighbours_count;};
    long long get_vector_core_part_neighbours_count(){return vector_core_part_neighbours_count;};
    long long get_collective_part_neighbours_count(){return collective_part_neighbours_count;};

    /* Print API */
    void print_stats();
    void print();

    /* frontier modification API */
    inline void set_all_active();
    inline void add_vertex(int src_id);
    inline void add_group_of_vertices(int *_vertex_ids, int _number_of_vertices);

    friend class GraphAbstractionsMulticore;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "frontier_multicore.hpp"
#include "modification.hpp"
#include "print.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
