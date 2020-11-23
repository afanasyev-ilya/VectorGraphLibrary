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
    auto sum_connections = [](int src_id, int connections_count, int vector_index)->int
    {
        return connections_count;
    };
    return this->reduce<int>(_graph, _frontier, sum_connections, REDUCE_SUM);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

