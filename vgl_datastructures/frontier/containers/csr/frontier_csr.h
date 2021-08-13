#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class FrontierCSR : public BaseFrontier
{
private:
    void init();

    #ifdef __USE_CSR_VERTEX_GROUPS__
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
    #endif

    #ifdef __USE_CSR_VERTEX_GROUPS__
    void copy_vertex_group_info_from_graph();
    #endif

    #ifdef __USE_CSR_VERTEX_GROUPS__
    int get_size_of_vertex_groups();
    size_t get_neighbours_of_vertex_groups();
    #endif

    #ifdef __USE_CSR_VERTEX_GROUPS__
    template <typename CopyCond>
    void copy_vertex_group_info_from_graph_cond(CopyCond _cond);
    #endif
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

    #ifdef __USE_GPU__
    void move_to_host();
    void move_to_device();
    #endif

    friend class GraphAbstractionsMulticore;
    friend class GraphAbstractionsNEC;
    friend class GraphAbstractionsGPU;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "frontier_csr.hpp"
#include "modification.hpp"
#include "print.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
