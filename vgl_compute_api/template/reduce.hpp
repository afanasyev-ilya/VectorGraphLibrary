#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename ReduceOperation, typename GraphContainer, typename FrontierContainer>
void GraphAbstractionsTEMPLATE::reduce_worker_sum(GraphContainer &_graph,
                                                  FrontierContainer &_frontier,
                                                  ReduceOperation &&reduce_op,
                                                  _T &_result)
{
    throw "Error in GraphAbstractionsTEMPLATE::reduce_worker_sum : not implemented yet";
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


