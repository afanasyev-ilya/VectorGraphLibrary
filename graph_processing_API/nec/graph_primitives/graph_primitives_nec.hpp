#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
GraphPrimitivesNEC::GraphPrimitivesNEC(VectCSRGraph<_TVertexValue, _TEdgeWeight> &_graph)
{
    cout << "TODO me" << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
_TEdgeWeight* GraphPrimitivesNEC::get_collective_weights(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                         FrontierNEC<_TVertexValue, _TEdgeWeight> &_frontier)
{
    if(_frontier.type == ALL_ACTIVE_FRONTIER)
        return (_graph.get_last_vertices_ve_ptr())->get_adjacent_weights();
    if(_frontier.collective_part_type == SPARSE_FRONTIER)
        return _graph.get_adjacent_weights();
    if(_frontier.collective_part_type == DENSE_FRONTIER)
        return _graph.get_adjacent_weights();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
