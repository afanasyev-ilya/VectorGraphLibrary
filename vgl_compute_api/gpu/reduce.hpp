#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename ReduceOperation, typename GraphContainer, typename FrontierContainer>
void GraphAbstractionsGPU::reduce_worker_sum(GraphContainer &_graph,
                                                  FrontierContainer &_frontier,
                                                  ReduceOperation &&reduce_op,
                                                  _T &_result)
{
    throw "Error in GraphAbstractionsGPU::reduce_worker_sum : not implemented yet";
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


