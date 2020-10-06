#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
GraphAbstractionsNEC<_TVertexValue, _TEdgeWeight>::GraphAbstractionsNEC(VectCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                                        TraversalDirection _initial_traversal)
{
    processed_graph_ptr = &_graph;
    traversal_direction = _initial_traversal;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void GraphAbstractionsNEC<_TVertexValue, _TEdgeWeight>::change_traversal_direction(TraversalDirection _new_direction)
{
    traversal_direction = _new_direction;

    // TODO what other changes are required?
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
