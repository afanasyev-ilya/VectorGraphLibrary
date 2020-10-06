#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


GraphPrimitivesNEC::GraphPrimitivesNEC(VectCSRGraph &_graph)
{
    cout << "TODO me" << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


_TEdgeWeight* GraphPrimitivesNEC::get_collective_weights(ExtendedCSRGraph &_graph,
                                                         FrontierNEC &_frontier)
{
    if(_frontier.type == ALL_ACTIVE_FRONTIER)
        return (_graph.get_ve_ptr())->get_adjacent_weights();
    if(_frontier.collective_part_type == SPARSE_FRONTIER)
        return _graph.get_adjacent_weights();
    if(_frontier.collective_part_type == DENSE_FRONTIER)
        return _graph.get_adjacent_weights();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
