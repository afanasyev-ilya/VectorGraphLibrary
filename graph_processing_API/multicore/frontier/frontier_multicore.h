#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class FrontierMulticore : public Frontier
{
private:
    // this is how NEC frontier is represented
    int *ids;
    int *flags;

    void init();
public:
    /* constructors and destructors */
    FrontierMulticore(VectCSRGraph &_graph, TraversalDirection _direction = SCATTER);
    ~FrontierMulticore();

    /* Get API */
    int *get_flags() {return flags;};
    int *get_ids() {return ids;};

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
