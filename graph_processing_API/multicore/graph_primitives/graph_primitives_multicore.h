#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../frontier/frontier_multicore.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class GraphPrimitivesMulticore
{
private:

public:
    GraphPrimitivesMulticore() {};

    ~GraphPrimitivesMulticore() {};

    template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation>
    void advance(ExtendedCSRGraph &_graph,
                 FrontierMulticore &_frontier,
                 EdgeOperation &&edge_op);

    // performs user-defined "compute_op" operation for each element in the given frontier
    template <typename _TVertexValue, typename _TEdgeWeight, typename ComputeOperation>
    void compute(ExtendedCSRGraph &_graph, FrontierMulticore &_frontier, ComputeOperation compute_op);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_primitives_multicore.hpp"
#include "compute.hpp"
#include "advance.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
