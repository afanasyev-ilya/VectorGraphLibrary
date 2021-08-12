#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class FrontierEdgesList : public BaseFrontier
{
private:
    void init();
public:
    /* constructors and destructors */
    FrontierEdgesList(VGL_Graph &_graph, TraversalDirection _direction = SCATTER);
    ~FrontierEdgesList();

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

#include "frontier_edges_list.hpp"
#include "modification.hpp"
#include "print.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
