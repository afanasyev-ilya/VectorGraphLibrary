/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight, typename FilterCondition>
void GraphPrimitivesNEC::filter(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                FrontierNEC &_frontier,
                                FilterCondition &&filter_cond)
{
    _frontier.filter(_graph, filter_cond);
    _frontier.print_frontier_info();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
