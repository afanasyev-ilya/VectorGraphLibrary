/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FilterCondition, typename GraphContainer, typename FrontierContainer>
void GraphAbstractionsTEMPLATE::generate_new_frontier_worker(GraphContainer &_graph,
                                                             FrontierContainer &_frontier,
                                                             FilterCondition &&filter_cond)
{
    throw "Error in GraphAbstractionsTEMPLATE::generate_new_frontier_worker : not implemented yet";
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FilterCondition>
void GraphAbstractionsTEMPLATE::generate_new_frontier(VGL_Graph &_graph,
                                                      VGL_Frontier &_frontier,
                                                      FilterCondition &&filter_cond)
{
    common_generate_new_frontier(_graph, _frontier, filter_cond, this);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
