#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
GraphAbstractionsNEC<_TVertexValue, _TEdgeWeight>::GraphAbstractionsNEC(VectCSRGraph<_TVertexValue, _TEdgeWeight> &_graph)
{
    _processed_graph_ptr = &_graph;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
