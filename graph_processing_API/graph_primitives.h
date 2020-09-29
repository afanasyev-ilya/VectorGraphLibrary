#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "frontier.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class GraphPrimitives
{
private:

public:
    GraphPrimitives() {};

    ~GraphPrimitives() {};

    template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation>
    void advance(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                 Frontier *_frontier,
                 EdgeOperation &&edge_op) {};

    // performs user-defined "compute_op" operation for each element in the given frontier
    template <typename _TVertexValue, typename _TEdgeWeight, typename ComputeOperation>
    void compute(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                 Frontier *_frontier,
                 ComputeOperation compute_op) {};
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
