#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "vertex_group.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class FrontierGeneral : public BaseFrontier
{
private:
    void init();

    CSRVertexGroup large_degree;
    CSRVertexGroup degree_256_to_128;
    CSRVertexGroup degree_128_to_64;
    CSRVertexGroup degree_64_to_32;
    CSRVertexGroup degree_32_to_16;
    CSRVertexGroup degree_16_to_8;
    CSRVertexGroup degree_8_to_0;

    void fill_vertex_group_data();
    void create_vertices_group_array(CSRVertexGroup &_group_data, int _bottom, int _top);
public:
    /* constructors and destructors */
    FrontierGeneral(VGL_Graph &_graph, TraversalDirection _direction = SCATTER);
    ~ FrontierGeneral();

    /* Print API */
    void print_stats();
    void print();

    /* frontier modification API */
    inline void set_all_active();
    inline void add_vertex(int _src_id);
    inline void add_group_of_vertices(int *_vertex_ids, int _number_of_vertices);
    void clear() {sparsity_type = SPARSE_FRONTIER; this->size = 0; neighbours_count = 0; };

    friend class GraphAbstractionsMulticore;
    friend class GraphAbstractionsNEC;
    friend class GraphAbstractionsGPU;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "frontier_general.hpp"
#include "modification.hpp"
#include "print.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////