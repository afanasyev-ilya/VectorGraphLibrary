/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight, typename FilterCondition>
void GraphPrimitivesNEC::generate_new_frontier(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                               FrontierNEC &_frontier,
                                               FilterCondition &&filter_cond)
{
    const int ve_threshold = _graph.get_nec_vector_engine_threshold_vertex();
    const int vc_threshold = _graph.get_nec_vector_core_threshold_vertex();
    const int vertices_count = _graph.get_vertices_count();

    // fill flags and calculate new frontier size
    int vertices_in_frontier = 0;
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma _NEC ivdep
    #pragma omp parallel for schedule(static) reduction(+: vertices_in_frontier)
    for (int src_id = 0; src_id < _frontier.max_size; src_id++)
    {
        int new_flag = filter_cond(src_id);
        _frontier.flags[src_id] = new_flag;
        vertices_in_frontier += new_flag;
    }
    _frontier.current_size = vertices_in_frontier;

    // chose frontier representation
    if(_frontier.current_size == _frontier.max_size) // no checks required
    {
        _frontier.type = ALL_ACTIVE_FRONTIER;
    }
    else if(double(_frontier.current_size)/_frontier.max_size > FRONTIER_TYPE_CHANGE_THRESHOLD) // flags array
    {
        _frontier.type = DENSE_FRONTIER;
    }
    else // queue + flags for now
    {
        _frontier.type = SPARSE_FRONTIER;

        _frontier.vector_engine_part_size = sparse_copy_if(_frontier.flags, _frontier.ids, _frontier.work_buffer, _frontier.max_size, 0, ve_threshold);

        _frontier.vector_core_part_size = sparse_copy_if(_frontier.flags, &_frontier.ids[_frontier.vector_engine_part_size], _frontier.work_buffer, _frontier.max_size, ve_threshold, vc_threshold);

        _frontier.collective_part_size = sparse_copy_if(_frontier.flags, &_frontier.ids[_frontier.vector_core_part_size + _frontier.vector_engine_part_size], _frontier.work_buffer, _frontier.max_size, vc_threshold, vertices_count);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
