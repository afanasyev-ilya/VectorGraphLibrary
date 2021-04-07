#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename ReduceOperation>
_T GraphAbstractionsMulticore::reduce_sum(UndirectedCSRGraph &_graph,
                                    FrontierMulticore &_frontier,
                                    ReduceOperation &&reduce_op)
{
    LOAD_UNDIRECTED_CSR_GRAPH_DATA(_graph);

    _T reduce_result = 0.0;

    if(_frontier.type == ALL_ACTIVE_FRONTIER)
    {
        int frontier_size = _frontier.max_size;

        #pragma simd
        #pragma vector
        #pragma ivdep
        #pragma omp parallel for schedule(static) reduction(+: reduce_result) // TODO different type
        for(int src_id = 0; src_id < frontier_size; src_id++)
        {
            int connections_count = vertex_pointers[src_id + 1] - vertex_pointers[src_id];
            int vector_index = get_vector_index(src_id);
            _T val = reduce_op(src_id, connections_count, vector_index);
            reduce_result += val;
        }
    }
    else if(_frontier.type == DENSE_FRONTIER)
    {
        int frontier_size = _frontier.max_size;
        int *frontier_flags = _frontier.flags;

        #pragma simd
        #pragma vector
        #pragma ivdep
        #pragma omp parallel for schedule(static) reduction(+: reduce_result)
        for (int src_id = 0; src_id < frontier_size; src_id++)
        {
            if(frontier_flags[src_id] == IN_FRONTIER_FLAG)
            {
                int connections_count = vertex_pointers[src_id + 1] - vertex_pointers[src_id];
                int vector_index = get_vector_index(src_id);
                _T val = reduce_op(src_id, connections_count, vector_index);
                reduce_result += val;
            }
        }
    }
    else if(_frontier.type == SPARSE_FRONTIER)
    {
        int frontier_size = _frontier.current_size;
        int *frontier_ids = _frontier.ids;

        #pragma simd
        #pragma vector
        #pragma ivdep
        #pragma omp parallel for schedule(static) reduction(+: reduce_result)
        for (int frontier_pos = 0; frontier_pos < frontier_size; frontier_pos++)
        {
            int src_id = frontier_ids[frontier_pos];
            int connections_count = vertex_pointers[src_id + 1] - vertex_pointers[src_id];
            int vector_index = get_vector_index(src_id);
            _T val = reduce_op(src_id, connections_count, vector_index);
            reduce_result += val;
        }
    }


    return reduce_result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename ReduceOperation>
_T GraphAbstractionsMulticore::reduce(VectCSRGraph &_graph,
                                FrontierMulticore &_frontier,
                                ReduceOperation &&reduce_op,
                                REDUCE_TYPE _reduce_type)
{
    Timer tm;
    tm.start();

    if(_frontier.get_direction() != current_traversal_direction)
    {
        throw "Error in GraphAbstractionsMulticore::reduce : wrong frontier direction";
    }

    UndirectedCSRGraph *current_direction_graph;
    if(current_traversal_direction == SCATTER)
    {
        current_direction_graph = _graph.get_outgoing_graph_ptr();
    }
    else if(current_traversal_direction == GATHER)
    {
        current_direction_graph = _graph.get_incoming_graph_ptr();
    }

    if(_reduce_type == REDUCE_SUM)
    {
        return reduce_sum<_T>(*current_direction_graph, _frontier, reduce_op);
    }
    else
    {
        throw "Error in GraphPrimitivesNEC::reduce: non-sum reduce are currently unsupported";
        return 0;
    }

    tm.end();
    performance_stats.update_reduce_time(tm);
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_bandwidth_stats("Reduce", _frontier.size(), REDUCE_INT_ELEMENTS*sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

