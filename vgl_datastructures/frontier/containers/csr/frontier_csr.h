#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class FrontierCSR : public BaseFrontier
{
private:
    void init();
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

#include "frontier_csr.hpp"
#include "modification.hpp"
#include "print.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
