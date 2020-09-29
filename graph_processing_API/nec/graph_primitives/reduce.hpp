#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename _TVertexValue, typename _TEdgeWeight, typename ReduceOperation>
_T GraphPrimitivesNEC::reduce_sum(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                  FrontierNEC &_frontier,
                                  ReduceOperation &&reduce_op)
{
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    double t1 = omp_get_wtime();
    #endif

    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);
    const long long int *vertex_pointers = outgoing_ptrs;

    _T reduce_result = 0.0;

    int max_frontier_size = _frontier.max_size;
    if(_frontier.type == ALL_ACTIVE_FRONTIER)
    {
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vector
        #pragma _NEC ivdep
        #pragma omp parallel for schedule(static) reduction(+: reduce_result)
        for(int src_id = 0; src_id < max_frontier_size; src_id++)
        {
            int connections_count = vertex_pointers[src_id + 1] - vertex_pointers[src_id];
            int vector_index = get_vector_index(src_id);
            _T val = reduce_op(src_id, connections_count, vector_index);
            reduce_result += val;
        }
    }
    else if((_frontier.type == DENSE_FRONTIER) || (_frontier.type == SPARSE_FRONTIER)) // TODO FIX SPARSE
    {
        int *frontier_flags = _frontier.flags;
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vector
        #pragma _NEC ivdep
        #pragma omp parallel for schedule(static) reduction(+: reduce_result)
        for (int src_id = 0; src_id < max_frontier_size; src_id++)
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

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    double t2 = omp_get_wtime();
    INNER_WALL_TIME += t2 - t1;
    INNER_REDUCE_TIME += t2 - t1;
    double work = _frontier.size();
    cout << "reduce time: " << (t2 - t1)*1000.0 << " ms" << endl;
    cout << "reduce BW: " << sizeof(int)*(REDUCE_INT_ELEMENTS)*work/((t2-t1)*1e9) << " GB/s" << endl << endl;
    #endif

    return reduce_result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename _TVertexValue, typename _TEdgeWeight, typename ReduceOperation>
_T GraphPrimitivesNEC::reduce(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                              FrontierNEC &_frontier,
                              ReduceOperation &&reduce_op,
                              REDUCE_TYPE _reduce_type)
{
    if(_reduce_type == REDUCE_SUM)
    {
        return reduce_sum<_T>(_graph, _frontier, reduce_op);
    }
    else
    {
        return 0;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

