#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

GraphAbstractionsMulticore::GraphAbstractionsMulticore(VGL_Graph &_graph, TraversalDirection _initial_traversal)
{
    processed_graph_ptr = &_graph;
    current_traversal_direction = _initial_traversal;
}

/*
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

long long GraphAbstractionsMulticore::count_frontier_neighbours(VectCSRGraph &_graph,
                                                          FrontierMulticore &_frontier)
{
    auto sum_connections = []__VGL_COMPUTE_ARGS__->int
    {
        return connections_count;
    };
    return this->reduce<int>(_graph, _frontier, sum_connections, REDUCE_SUM);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

long long GraphAbstractionsMulticore::compute_process_shift(long long _shard_shift,
                                                      TraversalDirection _traversal,
                                                      int _storage,
                                                      long long _edges_count,
                                                      bool _outgoing_graph_is_stored)
{
    long long traversal_shift = 0;
    if(_outgoing_graph_is_stored)
        traversal_shift = _traversal * direction_shift;
    long long storage_shift = _storage * _edges_count;
    return _shard_shift + traversal_shift + storage_shift;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////*/

