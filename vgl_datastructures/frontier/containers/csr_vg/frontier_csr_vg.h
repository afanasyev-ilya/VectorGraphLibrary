#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class FrontierCSR_VG : public BaseFrontier
{
private:
    void init();

    CSRVertexGroup vertex_groups[CSR_VERTEX_GROUPS_NUM];
    void copy_vertex_group_info_from_graph();

    int get_size_of_vertex_groups();
    size_t get_neighbours_of_vertex_groups();
    void print_vertex_group_sizes();

    template <typename CopyCond>
    void copy_vertex_group_info_from_graph_cond(CopyCond _cond);
public:
    /* constructors and destructors */
    FrontierCSR_VG(VGL_Graph &_graph, TraversalDirection _direction = SCATTER);
    ~ FrontierCSR_VG();

    /* Print API */
    void print_stats();
    void print();

    /* frontier modification API */
    inline void set_all_active();
    inline void add_vertex(int _src_id);
    inline void add_group_of_vertices(int *_vertex_ids, int _number_of_vertices);
    void clear();

    #ifdef __USE_GPU__
    void move_to_host();
    void move_to_device();
    #endif

    friend class GraphAbstractionsMulticore;
    friend class GraphAbstractionsNEC;
    friend class GraphAbstractionsGPU;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "frontier_csr_vg.hpp"
#include "modification.hpp"
#include "print.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
