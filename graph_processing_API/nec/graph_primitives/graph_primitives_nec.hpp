#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
_TEdgeWeight* GraphPrimitivesNEC::get_collective_weights(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                         FrontierNEC &_frontier)
{
    if(_frontier.type() == SPARSE_FRONTIER)
        return _graph.get_outgoing_weights();
    else
        return (_graph.get_last_vertices_ve_ptr())->get_adjacent_weights();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
