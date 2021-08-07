#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

GraphAbstractionsNEC::GraphAbstractionsNEC(VGL_Graph &_graph, TraversalDirection _initial_traversal)
{
    processed_graph_ptr = &_graph;
    current_traversal_direction = _initial_traversal;
    //direction_shift = _graph.get_direction_shift();
    use_safe_stores = false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

GraphAbstractionsNEC::~GraphAbstractionsNEC()
{

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

long long GraphAbstractionsNEC::count_frontier_neighbours(VGL_Graph &_graph,
                                                          VGL_Frontier &_frontier)
{
    auto sum_connections = []__VGL_COMPUTE_ARGS__->int
    {
        return connections_count;
    };
    return this->reduce<int>(_graph, _frontier, sum_connections, REDUCE_SUM);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

