#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "vector_registers.h"
#include "frontier_nec.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class GraphPrimitivesNEC
{
private:

public:
    GraphPrimitivesNEC() {};

    ~GraphPrimitivesNEC() {};

    template <typename InitOperation>
    void init(int size, InitOperation init_op);

    template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation>
    void advance(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                 FrontierNEC &_frontier,
                 EdgeOperation edge_op);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_primitives_nec.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
