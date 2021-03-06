/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FilterCondition>
void GraphAbstractionsMulticore::estimate_sorted_frontier_part_size(FrontierMulticore &_frontier,
                                                                    long long *_vertex_pointers,
                                                                    int _first_vertex,
                                                                    int _last_vertex,
                                                                    FilterCondition &&filter_cond,
                                                                    int &_elements_count,
                                                                    long long &_neighbours_count)
{
    int flags_sum = 0;
    long long connections_sum = 0;
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma _NEC ivdep
    #pragma omp parallel for schedule(static) reduction(+: flags_sum, connections_sum)
    for (int src_id = _first_vertex; src_id < _last_vertex; src_id++)
    {
        int connections_count = _vertex_pointers[src_id + 1] - _vertex_pointers[src_id];
        int new_flag = filter_cond(src_id, connections_count);
        _frontier.flags[src_id] = new_flag;
        flags_sum += new_flag;
        if(new_flag == IN_FRONTIER_FLAG)
            connections_sum += connections_count;
    }

    _elements_count = flags_sum;
    _neighbours_count = connections_sum;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FilterCondition>
void GraphAbstractionsMulticore::generate_new_frontier(VectCSRGraph &_graph,
                                                       FrontierMulticore &_frontier,
                                                       FilterCondition &&filter_cond)
{
    Timer tm_flags, tm_wall;
    tm_flags.start();
    tm_wall.start();

    _frontier.set_direction(current_traversal_direction);

    UndirectedCSRGraph *current_direction_graph = _graph.get_direction_graph_ptr(current_traversal_direction);
    LOAD_UNDIRECTED_CSR_GRAPH_DATA((*current_direction_graph));

    const int ve_threshold = current_direction_graph->get_vector_engine_threshold_vertex();
    int vc_threshold = current_direction_graph->get_vector_core_threshold_vertex();

    // calculate numbers of elements in different frontier parts
    estimate_sorted_frontier_part_size(_frontier, vertex_pointers, 0, ve_threshold, filter_cond,
                          _frontier.vector_engine_part_size, _frontier.vector_engine_part_neighbours_count);
    estimate_sorted_frontier_part_size(_frontier, vertex_pointers, ve_threshold, vc_threshold, filter_cond,
                          _frontier.vector_core_part_size, _frontier.vector_core_part_neighbours_count);
    estimate_sorted_frontier_part_size(_frontier, vertex_pointers, vc_threshold, vertices_count, filter_cond,
                          _frontier.collective_part_size, _frontier.collective_part_neighbours_count);

    // calculate total size of frontier
    _frontier.current_size = _frontier.vector_engine_part_size + _frontier.vector_core_part_size + _frontier.collective_part_size;
    _frontier.neighbours_count = _frontier.vector_engine_part_neighbours_count + _frontier.vector_core_part_neighbours_count + _frontier.collective_part_neighbours_count;

    tm_flags.end();
    Timer tm_copy_if;
    tm_copy_if.start();

    bool copy_if_work = false;

    // set type of the whole frontier
    if(_frontier.current_size == _frontier.max_size)
    {
        _frontier.type = ALL_ACTIVE_FRONTIER;
    }
    else if(double(_frontier.current_size)/_frontier.max_size > 0.7) // flags array
    {
        _frontier.type = DENSE_FRONTIER;
        _frontier.vector_engine_part_type = DENSE_FRONTIER;
        _frontier.vector_core_part_type = DENSE_FRONTIER;
        _frontier.collective_part_type = DENSE_FRONTIER;
    }
    else
    {
        copy_if_work = true;
        _frontier.type = SPARSE_FRONTIER;
        _frontier.vector_engine_part_type = SPARSE_FRONTIER;
        _frontier.vector_core_part_type = SPARSE_FRONTIER;
        _frontier.collective_part_type = SPARSE_FRONTIER;
        parallel_buffers_copy_if(_frontier.flags,  _frontier.ids, _frontier.work_buffer, _frontier.max_size);
    }

    tm_copy_if.end();
    tm_wall.end();
    long long work = _frontier.max_size;
    performance_stats.update_gnf_time(tm_wall);
    performance_stats.update_bytes_requested(work*2.0*sizeof(int));
    if(copy_if_work)
        performance_stats.update_bytes_requested(work*2.0*sizeof(int));

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm_flags.print_bandwidth_stats("GNF flags", work, 2.0*sizeof(int));
    if(copy_if_work)
        tm_copy_if.print_bandwidth_stats("GNF copy if", work, 2.0*sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
