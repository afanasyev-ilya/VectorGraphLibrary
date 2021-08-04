#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class FrontierGeneral : public BaseFrontier
{
private:
    // this is how General frontier is represented
    int *work_buffer;

    int neighbours_count;

    void init();
public:
    /* constructors and destructors */
    FrontierGeneral(VGL_Graph &_graph, TraversalDirection _direction = SCATTER);
    ~ FrontierGeneral();

    /* Get API */


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
