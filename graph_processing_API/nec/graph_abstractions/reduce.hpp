#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename ReduceOperation>
_T GraphAbstractionsNEC::reduce_worker_sum(VGL_Graph &_graph,
                                           VGL_Frontier &_frontier,
                                           ReduceOperation &&reduce_op)
{
    UndirectedGraph *current_direction_graph = _graph.get_direction_data(current_traversal_direction);

    int frontier_size = _frontier.get_size();
    int *frontier_flags = _frontier.get_flags();
    int *frontier_ids = _frontier.get_ids();
    FrontierSparsityType frontier_type = _frontier.get_sparsity_type();

    _T reduce_result = 0.0;

    if(_frontier.get_sparsity_type() == ALL_ACTIVE_FRONTIER)
    {
        #pragma _NEC cncall
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vector
        #pragma omp parallel for schedule(static) reduction(+: reduce_result)
        for(int src_id = 0; src_id < frontier_size; src_id++)
        {
            int connections_count = current_direction_graph->get_connections_count(src_id);
            int vector_index = get_vector_index(src_id);
            _T val = reduce_op(src_id, connections_count, vector_index);
            reduce_result += val;
        }
    }
    else if(_frontier.get_sparsity_type() == DENSE_FRONTIER)
    {
        #pragma _NEC cncall
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vector
        #pragma omp parallel for schedule(static) reduction(+: reduce_result)
        for (int src_id = 0; src_id < frontier_size; src_id++)
        {
            if(frontier_flags[src_id] == IN_FRONTIER_FLAG)
            {
                int connections_count = current_direction_graph->get_connections_count(src_id);
                int vector_index = get_vector_index(src_id);
                _T val = reduce_op(src_id, connections_count, vector_index);
                reduce_result += val;
            }
        }
    }
    else if(_frontier.get_sparsity_type() == SPARSE_FRONTIER)
    {
        #pragma _NEC cncall
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vector
        #pragma omp parallel for schedule(static) reduction(+: reduce_result)
        for (int frontier_pos = 0; frontier_pos < frontier_size; frontier_pos++)
        {
            int src_id = frontier_ids[frontier_pos];
            int connections_count = current_direction_graph->get_connections_count(src_id);
            int vector_index = get_vector_index(src_id);
            _T val = reduce_op(src_id, connections_count, vector_index);
            reduce_result += val;
        }
    }

    return reduce_result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename ReduceOperation>
_T GraphAbstractionsNEC::reduce(VGL_Graph &_graph,
                                VGL_Frontier &_frontier,
                                ReduceOperation &&reduce_op,
                                REDUCE_TYPE _reduce_type)
{
    Timer tm;
    tm.start();

    _T reduce_result = 0;

    if(_frontier.get_direction() != current_traversal_direction)
    {
        throw "Error in GraphAbstractionsNEC::reduce : wrong frontier direction";
    }

    if(omp_in_parallel())
    {
        throw "Error in GraphAbstractionsNEC::reduce : reduce can not be called in parallel region (reduction construct)";
    }

    if(_reduce_type == REDUCE_SUM)
    {
        reduce_result = reduce_worker_sum<_T>(_graph, _frontier, reduce_op);
    }
    else
    {
        throw "Error in GraphAbstractionsNEC::reduce: non-sum reduce are currently unsupported";
    }

    tm.end();
    long long work = _frontier.size();
    performance_stats.update_reduce_time(tm);
    performance_stats.update_bytes_requested(REDUCE_INT_ELEMENTS*sizeof(int)*work);
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_bandwidth_stats("Reduce", work, REDUCE_INT_ELEMENTS*sizeof(int));
    #endif

    return reduce_result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

