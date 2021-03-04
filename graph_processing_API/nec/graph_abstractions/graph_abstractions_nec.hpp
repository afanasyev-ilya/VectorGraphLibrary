#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

GraphAbstractionsNEC::GraphAbstractionsNEC(VectCSRGraph &_graph, TraversalDirection _initial_traversal)
{
    processed_graph_ptr = &_graph;
    current_traversal_direction = _initial_traversal;
    direction_shift = _graph.get_direction_shift();
    use_safe_stores = false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

GraphAbstractionsNEC::GraphAbstractionsNEC(ShardedCSRGraph &_graph, TraversalDirection _initial_traversal)
{
    processed_graph_ptr = &_graph;
    current_traversal_direction = _initial_traversal;
    direction_shift = _graph.get_direction_shift();
    use_safe_stores = false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

GraphAbstractionsNEC::GraphAbstractionsNEC(EdgesListGraph &_graph, TraversalDirection _initial_traversal)
{
    processed_graph_ptr = &_graph;
    current_traversal_direction = _initial_traversal;
    direction_shift = 0;
    use_safe_stores = false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

long long GraphAbstractionsNEC::count_frontier_neighbours(VectCSRGraph &_graph,
                                                          FrontierNEC &_frontier)
{
    auto sum_connections = []__VGL_COMPUTE_ARGS__->int
    {
        return connections_count;
    };
    return this->reduce<int>(_graph, _frontier, sum_connections, REDUCE_SUM);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

long long GraphAbstractionsNEC::compute_process_shift(long long _shard_shift,
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

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

