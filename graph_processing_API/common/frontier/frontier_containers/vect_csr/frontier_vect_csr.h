#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class FrontierVectorCSR : public BaseFrontier
{
private:
    // this is how Vector CSR frontier is represented
    int *work_buffer;

    int neighbours_count;

    int vector_engine_part_size;
    int vector_core_part_size;
    int collective_part_size;

    long long vector_engine_part_neighbours_count;
    long long vector_core_part_neighbours_count;
    long long collective_part_neighbours_count;

    FrontierSparsityType vector_engine_part_type;
    FrontierSparsityType vector_core_part_type;
    FrontierSparsityType collective_part_type;

    void init();
public:
    /* constructors and destructors */
    FrontierVectorCSR(VGL_Graph &_graph, TraversalDirection _direction = SCATTER);
    ~FrontierVectorCSR();

    /* Get API */
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
    inline void add_vertex(int _src_id);
    inline void add_group_of_vertices(int *_vertex_ids, int _number_of_vertices);
    void clear() {sparsity_type = SPARSE_FRONTIER; this->size = 0; neighbours_count = 0; };
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "frontier_vect_csr.hpp"
#include "modification.hpp"
#include "print.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
