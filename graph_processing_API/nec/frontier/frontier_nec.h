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

    long long vector_engine_part_neighbours_count;
    long long vector_core_part_neighbours_count;
    long long collective_part_neighbours_count;

    FrontierType vector_engine_part_type;
    FrontierType vector_core_part_type;
    FrontierType collective_part_type;

    void init();
public:
    /* constructors and destructors */
    FrontierNEC(VectCSRGraph &_graph, TraversalDirection _direction = SCATTER);
    FrontierNEC(ShardedCSRGraph &_graph, TraversalDirection _direction = SCATTER);
    ~FrontierNEC();

    /* Get API */
    inline int *get_flags() {return flags;};
    inline int *get_ids() {return ids;};
    inline int get_vector_engine_part_size(){return vector_engine_part_size;};
    inline int get_vector_core_part_size(){return vector_core_part_size;};
    inline int get_collective_part_size(){return collective_part_size;};

    inline long long get_vector_engine_part_neighbours_count(){return vector_engine_part_neighbours_count;};
    inline long long get_vector_core_part_neighbours_count(){return vector_core_part_neighbours_count;};
    inline long long get_collective_part_neighbours_count(){return collective_part_neighbours_count;};

    /* Print API */
    void print_stats();
    void print();

    /* frontier modification API */
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
