/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FilterCondition>
int GraphAbstractionsNEC::estimate_sorted_frontier_part_size(FrontierNEC &_frontier,
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

template <typename FilterCondition>
void GraphAbstractionsNEC::generate_new_frontier(VectCSRGraph &_graph,
                                                 FrontierNEC &_frontier,
                                                 FilterCondition &&filter_cond)
{
    Timer tm_flags;
    tm_flags.start();

    _frontier.set_direction(current_traversal_direction);

    UndirectedCSRGraph *graph_ptr = _graph.get_direction_graph_ptr(current_traversal_direction);
    const int ve_threshold = graph_ptr->get_vector_engine_threshold_vertex();
    int vc_threshold = graph_ptr->get_vector_core_threshold_vertex();
    const int vertices_count = graph_ptr->get_vertices_count();

    // calculate numbers of elements in different frontier parts
    _frontier.vector_engine_part_size = estimate_sorted_frontier_part_size(_frontier, 0, ve_threshold, filter_cond);
    _frontier.vector_core_part_size = estimate_sorted_frontier_part_size(_frontier, ve_threshold, vc_threshold, filter_cond);
    _frontier.collective_part_size = estimate_sorted_frontier_part_size(_frontier, vc_threshold, vertices_count, filter_cond);

    // calculate total size of frontier
    _frontier.current_size = _frontier.vector_engine_part_size + _frontier.vector_core_part_size + _frontier.collective_part_size;

    tm_flags.end();
    Timer tm_copy_if;
    tm_copy_if.start();

    // set type of the whole frontier
    if(_frontier.current_size == _frontier.max_size)
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

    // estimate first (VE) part sparsity
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

    // estimate second (VC) part sparsity
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

    // estimate third (collective) part sparsity
    if(double(_frontier.collective_part_size)/(vertices_count - vc_threshold) < COLLECTIVE_FRONTIER_TYPE_CHANGE_THRESHOLD)
    {
        _frontier.collective_part_type = SPARSE_FRONTIER;
        if(_frontier.collective_part_size > 0)
        {

            int segment_shift = vc_threshold;
            int segment_size = vertices_count - vc_threshold;
            int copied_elements = dense_copy_if(&_frontier.flags[segment_shift], &_frontier.ids[_frontier.vector_core_part_size + _frontier.vector_engine_part_size], _frontier.work_buffer, segment_size, segment_shift, SAVE_ORDER);
        }
    }
    else
    {
        _frontier.collective_part_type = DENSE_FRONTIER;
    }

    tm_copy_if.end();
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    INNER_WALL_TIME += tm_flags.get_time() + tm_copy_if.get_time();
    INNER_GNF_TIME += tm_flags.get_time() + tm_copy_if.get_time();
    tm_flags.print_time_and_bandwidth_stats("GNF flags", _frontier.max_size, 2.0*sizeof(int));
    tm_copy_if.print_time_and_bandwidth_stats("GNF copy if", _frontier.max_size, 2.0*sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
