#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename ReduceOperation, typename GraphContainer, typename FrontierContainer>
void GraphAbstractionsMulticore::reduce_worker_sum(GraphContainer &_graph,
                                                   FrontierContainer &_frontier,
                                                   ReduceOperation &&reduce_op,
                                                   _T &_result)
{
    int frontier_size = _frontier.get_size();
    int *frontier_flags = _frontier.get_flags();
    int *frontier_ids = _frontier.get_ids();
    FrontierSparsityType frontier_type = _frontier.get_sparsity_type();

    _T reduce_result = 0.0;

    if(_frontier.get_sparsity_type() == ALL_ACTIVE_FRONTIER)
    {
        #pragma simd
        #pragma vector
        #pragma ivdep
        #pragma omp parallel for schedule(static) reduction(+: reduce_result)
        for(int src_id = 0; src_id < frontier_size; src_id++)
        {
            int connections_count = _graph.get_connections_count(src_id);
            int vector_index = get_vector_index(src_id);
            _T val = reduce_op(src_id, connections_count, vector_index);
            reduce_result += val;
        }
    }
    else if(_frontier.get_sparsity_type() == DENSE_FRONTIER)
    {
        #pragma simd
        #pragma vector
        #pragma ivdep
        #pragma omp parallel for schedule(static) reduction(+: reduce_result)
        for (int src_id = 0; src_id < frontier_size; src_id++)
        {
            if(frontier_flags[src_id] == IN_FRONTIER_FLAG)
            {
                int connections_count = _graph.get_connections_count(src_id);
                int vector_index = get_vector_index(src_id);
                _T val = reduce_op(src_id, connections_count, vector_index);
                reduce_result += val;
            }
        }
    }
    else if(_frontier.get_sparsity_type() == SPARSE_FRONTIER)
    {
        #pragma simd
        #pragma vector
        #pragma ivdep
        #pragma omp parallel for schedule(static) reduction(+: reduce_result)
        for (int frontier_pos = 0; frontier_pos < frontier_size; frontier_pos++)
        {
            int src_id = frontier_ids[frontier_pos];
            int connections_count = _graph.get_connections_count(src_id);
            int vector_index = get_vector_index(src_id);
            _T val = reduce_op(src_id, connections_count, vector_index);
            reduce_result += val;
        }
    }

    _result = reduce_result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename ReduceOperation, typename GraphContainer, typename FrontierContainer>
void GraphAbstractionsMulticore::reduce_worker_max(GraphContainer &_graph,
                                                   FrontierContainer &_frontier,
                                                   ReduceOperation &&reduce_op,
                                                   _T &_result)
{
    int frontier_size = _frontier.get_size();
    int *frontier_flags = _frontier.get_flags();
    int *frontier_ids = _frontier.get_ids();
    FrontierSparsityType frontier_type = _frontier.get_sparsity_type();

    _T reduce_result = 0.0;

    if(_frontier.get_sparsity_type() == ALL_ACTIVE_FRONTIER)
    {
        #pragma simd
        #pragma vector
        #pragma ivdep
        #pragma omp parallel for schedule(static) reduction(+: reduce_result)
        for(int src_id = 0; src_id < frontier_size; src_id++)
        {
            int connections_count = _graph.get_connections_count(src_id);
            int vector_index = get_vector_index(src_id);
            _T val = reduce_op(src_id, connections_count, vector_index);
            if(val > reduce_result)
                reduce_result = val;
        }
    }
    else if(_frontier.get_sparsity_type() == DENSE_FRONTIER)
    {
        #pragma simd
        #pragma vector
        #pragma ivdep
        #pragma omp parallel for schedule(static) reduction(+: reduce_result)
        for (int src_id = 0; src_id < frontier_size; src_id++)
        {
            if(frontier_flags[src_id] == IN_FRONTIER_FLAG)
            {
                int connections_count = _graph.get_connections_count(src_id);
                int vector_index = get_vector_index(src_id);
                _T val = reduce_op(src_id, connections_count, vector_index);
                if(val > reduce_result)
                    reduce_result = val;
            }
        }
    }
    else if(_frontier.get_sparsity_type() == SPARSE_FRONTIER)
    {
        #pragma simd
        #pragma vector
        #pragma ivdep
        #pragma omp parallel for schedule(static) reduction(+: reduce_result)
        for (int frontier_pos = 0; frontier_pos < frontier_size; frontier_pos++)
        {
            int src_id = frontier_ids[frontier_pos];
            int connections_count = _graph.get_connections_count(src_id);
            int vector_index = get_vector_index(src_id);
            _T val = reduce_op(src_id, connections_count, vector_index);
            if(val > reduce_result)
                reduce_result = val;
        }
    }

    _result = reduce_result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename ReduceOperation, typename GraphContainer, typename FrontierContainer>
void GraphAbstractionsMulticore::reduce_worker(GraphContainer &_graph,
                                               FrontierContainer &_frontier,
                                               ReduceOperation &&reduce_op,
                                               REDUCE_TYPE _reduce_type,
                                               _T &_result)
{
    if(_reduce_type == REDUCE_SUM)
        reduce_worker_sum(_graph, _frontier, reduce_op, _result);
    else if(_reduce_type == REDUCE_MAX)
        reduce_worker_max(_graph, _frontier, reduce_op, _result);
    else
        throw "Error in GraphAbstractionsNEC::reduce_worker: unsupported reduce type";
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename ReduceOperation>
_T GraphAbstractionsMulticore::reduce(VGL_Graph &_graph,
                                      VGL_Frontier &_frontier,
                                      ReduceOperation &&reduce_op,
                                      REDUCE_TYPE _reduce_type)
{
    _T result = 0;
    this->common_reduce(_graph, _frontier, reduce_op, _reduce_type, result, this);
    return result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

