#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "vertex_group.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class FrontierCSR : public BaseFrontier
{
private:
    void init();

    #ifndef __USE_GPU__
    CSRVertexGroup large_degree;
    CSRVertexGroup degree_128_256;
    CSRVertexGroup degree_64_128;
    CSRVertexGroup degree_32_64;
    CSRVertexGroup degree_16_32;
    CSRVertexGroup degree_8_16;
    CSRVertexGroup degree_0_8;
    #else
    CSRVertexGroup large_degree;
    CSRVertexGroup degree_32_1024;
    CSRVertexGroup degree_16_32;
    CSRVertexGroup degree_8_16;
    CSRVertexGroup degree_4_8;
    CSRVertexGroup degree_0_4;
    #endif

    void fill_vertex_group_data();
    void create_vertices_group_array(CSRVertexGroup &_group_data, int _bottom, int _top);
public:
    /* constructors and destructors */
    FrontierCSR(VGL_Graph &_graph, TraversalDirection _direction = SCATTER);
    ~ FrontierCSR();

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

#include "frontier_csr.hpp"
#include "modification.hpp"
#include "print.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////