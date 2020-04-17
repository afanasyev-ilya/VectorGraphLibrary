/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FilterCondition>
int GraphPrimitivesNEC::estimate_sorted_frontier_part_size(FrontierNEC &_frontier,
                                                           int _first_vertex,
                                                           int _last_vertex,
                                                           FilterCondition &&filter_cond)
{
    int flags_sum = 0;
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma _NEC ivdep
    #pragma omp parallel for schedule(static) reduction(+: flags_sum)
    for (int src_id = _first_vertex; src_id < _last_vertex; src_id++)
    {
        int new_flag = filter_cond(src_id);
        _frontier.flags[src_id] = new_flag;
        flags_sum += new_flag;
    }

    return flags_sum;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight, typename FilterCondition>
void GraphPrimitivesNEC::generate_new_frontier(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                               FrontierNEC &_frontier,
                                               FilterCondition &&filter_cond)
{
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    double t1 = omp_get_wtime();
    #endif

    const int ve_threshold = _graph.get_nec_vector_engine_threshold_vertex();
    const int vc_threshold = _graph.get_nec_vector_core_threshold_vertex();
    const int vertices_count = _graph.get_vertices_count();

    _frontier.vector_engine_part_size = estimate_sorted_frontier_part_size(_frontier, 0, ve_threshold, filter_cond);
    _frontier.vector_core_part_size = estimate_sorted_frontier_part_size(_frontier, ve_threshold, vc_threshold, filter_cond);
    _frontier.collective_part_size = estimate_sorted_frontier_part_size(_frontier, vc_threshold, vertices_count, filter_cond);
    _frontier.current_size = _frontier.vector_engine_part_size + _frontier.vector_core_part_size + _frontier.collective_part_size;

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    double t2 = omp_get_wtime();
    #endif

    if(_frontier.current_size == _frontier.max_size) // no checks required
    {
        _frontier.type = ALL_ACTIVE_FRONTIER;
    }
    else if(double(_frontier.current_size)/_frontier.max_size > FRONTIER_TYPE_CHANGE_THRESHOLD) // flags array
    {
        _frontier.type = DENSE_FRONTIER;
    }
    else
    {
        _frontier.type = SPARSE_FRONTIER;
    }

    if(double(_frontier.vector_engine_part_size)/(ve_threshold - 0) < VE_FRONTIER_TYPE_CHANGE_THRESHOLD)
    {
        _frontier.vector_engine_part_type = SPARSE_FRONTIER;
        if(_frontier.vector_engine_part_size > 0)
            sparse_copy_if(_frontier.flags, _frontier.ids, _frontier.work_buffer, _frontier.max_size, 0, ve_threshold);
    }
    else
    {
        _frontier.vector_engine_part_type = DENSE_FRONTIER;
    }

    if(double(_frontier.vector_core_part_size)/(vc_threshold - ve_threshold) < VC_FRONTIER_TYPE_CHANGE_THRESHOLD)
    {
        _frontier.vector_core_part_type = SPARSE_FRONTIER;
        if(_frontier.vector_core_part_size > 0)
            sparse_copy_if(_frontier.flags, &_frontier.ids[_frontier.vector_engine_part_size], _frontier.work_buffer, _frontier.max_size, ve_threshold, vc_threshold);
    }
    else
    {
        _frontier.vector_core_part_type = DENSE_FRONTIER;
    }

    if(double(_frontier.collective_part_size)/(vertices_count - vc_threshold) < COLLECTIVE_FRONTIER_TYPE_CHANGE_THRESHOLD)
    {
        _frontier.collective_part_type = SPARSE_FRONTIER;
        if(_frontier.collective_part_size > 0)
            sparse_copy_if(_frontier.flags, &_frontier.ids[_frontier.vector_core_part_size + _frontier.vector_engine_part_size], _frontier.work_buffer, _frontier.max_size, vc_threshold, vertices_count);
    }
    else
    {
        _frontier.collective_part_type = DENSE_FRONTIER;
    }

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    double t3 = omp_get_wtime();
    INNER_WALL_NEC_TIME += t3 - t1;
    INNER_GNF_NEC_TIME += t3 - t1;

    cout << "GNF flags time: " << 1000*(t2 - t1) << " ms" << endl;
    cout << "GNF flags BW: " << 2.0*sizeof(int)*_frontier.max_size/((t2-t1)*1e9) << " GB/s" << endl;
    cout << "GNF copy if time: " << 1000*(t3 - t2) << " ms" << endl;
    cout << "GNF copy if BW: " << 2.0*sizeof(int)*_frontier.max_size/((t3-t2)*1e9) << " GB/s" << endl << endl;
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
