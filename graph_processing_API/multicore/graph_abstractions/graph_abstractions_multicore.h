#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class GraphAbstractionsMulticore : public GraphAbstractions
{
private:

public:
    /*template <typename EdgeOperation>
    void advance(UndirectedCSRGraph &_graph,
                 FrontierMulticore &_frontier,
                 EdgeOperation &&edge_op);*/

    // performs user-defined "compute_op" operation for each element in the given frontier
    template <typename ComputeOperation>
    void compute(VectCSRGraph &_graph, FrontierMulticore &_frontier, ComputeOperation compute_op);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_abstractions_multicore.hpp"
#include "compute.hpp"
//#include "advance.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
