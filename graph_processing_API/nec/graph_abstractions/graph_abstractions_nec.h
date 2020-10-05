#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum TraversalDirection {
    SCATTER_TRAVERSAL = 0,
    GATHER_TRAVERSAL = 1
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
class GraphAbstractionsNEC
{
private:
    VectCSRGraph<_TVertexValue, _TEdgeWeight> *_processed_graph_ptr;
    TraversalDirection _traversal_direction;

    template <typename ComputeOperation>
    void compute_worker(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                        FrontierNEC<_TVertexValue, _TEdgeWeight> &_frontier,
                        ComputeOperation &&compute_op);
public:
    GraphAbstractionsNEC(VectCSRGraph<_TVertexValue, _TEdgeWeight> &_graph);

    template <typename ComputeOperation>
    void compute(VectCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                 FrontierNEC<_TVertexValue, _TEdgeWeight> &_frontier,
                 ComputeOperation &&compute_op);
};


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_abstractions_nec.hpp"
#include "compute.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
