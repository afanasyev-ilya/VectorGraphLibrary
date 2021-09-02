/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FilterCondition>
void GraphAbstractionsNEC::estimate_sorted_frontier_part_size(FrontierVectorCSR &_frontier,
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
void GraphAbstractionsNEC::generate_new_frontier_worker(EdgesListGraph &_graph,
                                                        FrontierEdgesList &_frontier,
                                                        FilterCondition &&filter_cond)
{
    Timer tm_wall;
    tm_wall.start();

    _frontier.set_direction(current_traversal_direction);
    int vertices_count = _graph.get_vertices_count();
    int *frontier_flags = _frontier.flags;
    int *frontier_ids = _frontier.ids;

    int elements_count = 0;
    long long neighbours_count = 0;

    #pragma _NEC cncall
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma _NEC ivdep
    #pragma omp parallel for schedule(static) reduction(+: elements_count, neighbours_count)
    for (int src_id = 0; src_id < vertices_count; src_id++)
    {
        int connections_count = 0;//_graph.get_connections_count(src_id);
        int new_flag = filter_cond(src_id, connections_count);
        frontier_flags[src_id] = new_flag;
        elements_count += new_flag;
        if(new_flag == IN_FRONTIER_FLAG)
            neighbours_count += connections_count;
    }

    _frontier.size = elements_count;
    _frontier.neighbours_count = neighbours_count;

    tm_wall.end();
    long long work = vertices_count;
    performance_stats.update_gnf_time(tm_wall);
    performance_stats.update_bytes_requested(work*2.0*sizeof(int));

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm_wall.print_bandwidth_stats("GNF", vertices_count, 2.0*sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FilterCondition>
void GraphAbstractionsNEC::generate_new_frontier_worker(CSRGraph &_graph,
                                                        FrontierCSR &_frontier,
                                                        FilterCondition &&filter_cond)
{
    Timer tm_wall;
    tm_wall.start();

    _frontier.set_direction(current_traversal_direction);
    int vertices_count = _graph.get_vertices_count();
    int *frontier_flags = _frontier.flags;
    int *frontier_ids = _frontier.ids;

    int elements_count = 0;
    long long neighbours_count = 0;

    #pragma _NEC cncall
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma _NEC ivdep
    #pragma omp parallel for schedule(static) reduction(+: elements_count, neighbours_count)
    for (int src_id = 0; src_id < vertices_count; src_id++)
    {
        int connections_count = _graph.get_connections_count(src_id);
        int new_flag = filter_cond(src_id, connections_count);
        frontier_flags[src_id] = new_flag;
        elements_count += new_flag;
        if(new_flag == IN_FRONTIER_FLAG)
            neighbours_count += connections_count;
    }

    auto in_frontier = [frontier_flags] (int src_id) {
        return frontier_flags[src_id];
    };
    _frontier.neighbours_count = neighbours_count;
    _frontier.size = ParallelPrimitives::copy_if_indexes(in_frontier, frontier_ids, vertices_count, _frontier.work_buffer, vertices_count, 0);

    tm_wall.end();
    long long work = vertices_count;
    performance_stats.update_gnf_time(tm_wall);
    performance_stats.update_bytes_requested(work*4.0*sizeof(int));

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm_wall.print_bandwidth_stats("GNF", vertices_count, 4.0*sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FilterCondition>
void GraphAbstractionsNEC::generate_new_frontier_worker(CSR_VG_Graph &_graph,
                                                        FrontierCSR_VG &_frontier,
                                                        FilterCondition &&filter_cond)
{
    Timer tm_wall;
    tm_wall.start();

    _frontier.set_direction(current_traversal_direction);
    int vertices_count = _graph.get_vertices_count();
    int *frontier_flags = _frontier.flags;
    int *frontier_ids = _frontier.ids;

    int elements_count = 0;
    long long neighbours_count = 0;

    #pragma _NEC cncall
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma _NEC ivdep
    #pragma omp parallel for schedule(static) reduction(+: elements_count, neighbours_count)
    for (int src_id = 0; src_id < vertices_count; src_id++)
    {
        int connections_count = _graph.get_connections_count(src_id);
        int new_flag = filter_cond(src_id, connections_count);
        frontier_flags[src_id] = new_flag;
        elements_count += new_flag;
        if(new_flag == IN_FRONTIER_FLAG)
            neighbours_count += connections_count;
    }

    auto filter_vertex_group = [frontier_flags] (int _src_id)->int {
        return frontier_flags[_src_id];
    };
    _frontier.copy_vertex_group_info_from_graph_cond(filter_vertex_group);

    int copy_pos = 0;
    for(int i = 0; i < CSR_VERTEX_GROUPS_NUM; i++)
    {
        if(_frontier.vertex_groups[i].size > 0)
        {
            memcpy(frontier_ids + copy_pos, _frontier.vertex_groups[i].ids, _frontier.vertex_groups[i].size * sizeof(int));
            copy_pos += _frontier.vertex_groups[i].size;
        }
    }
    _frontier.size = copy_pos;

    tm_wall.end();
    long long work = vertices_count;
    performance_stats.update_gnf_time(tm_wall);
    performance_stats.update_bytes_requested(work*4.0*sizeof(int));

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm_wall.print_bandwidth_stats("GNF", vertices_count, 4.0*sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FilterCondition>
void GraphAbstractionsNEC::generate_new_frontier_worker(VectorCSRGraph &_graph,
                                                        FrontierVectorCSR &_frontier,
                                                        FilterCondition &&filter_cond)
{
    Timer tm_flags, tm_wall;
    tm_flags.start();
    tm_wall.start();

    _frontier.set_direction(current_traversal_direction);

    LOAD_VECTOR_CSR_GRAPH_DATA(_graph);

    const int ve_threshold = _graph.get_vector_engine_threshold_vertex();
    int vc_threshold = _graph.get_vector_core_threshold_vertex();

    // calculate numbers of elements in different frontier parts
    estimate_sorted_frontier_part_size(_frontier, vertex_pointers, 0, ve_threshold, filter_cond,
                          _frontier.vector_engine_part_size, _frontier.vector_engine_part_neighbours_count);
    estimate_sorted_frontier_part_size(_frontier, vertex_pointers, ve_threshold, vc_threshold, filter_cond,
                          _frontier.vector_core_part_size, _frontier.vector_core_part_neighbours_count);
    estimate_sorted_frontier_part_size(_frontier, vertex_pointers, vc_threshold, vertices_count, filter_cond,
                          _frontier.collective_part_size, _frontier.collective_part_neighbours_count);

    // calculate total size of frontier
    _frontier.size = _frontier.vector_engine_part_size + _frontier.vector_core_part_size + _frontier.collective_part_size;
    _frontier.neighbours_count = _frontier.vector_engine_part_neighbours_count + _frontier.vector_core_part_neighbours_count + _frontier.collective_part_neighbours_count;

    tm_flags.end();
    Timer tm_copy_if;
    tm_copy_if.start();

    int *frontier_flags = _frontier.flags;
    auto in_frontier = [frontier_flags] (int src_id) {
        return frontier_flags[src_id];
    };

    bool copy_if_work = false;
    // estimate first (VE) part sparsity
    if(double(_frontier.vector_engine_part_size)/(ve_threshold - 0) < VE_FRONTIER_TYPE_CHANGE_THRESHOLD)
    {
        _frontier.vector_engine_part_type = SPARSE_FRONTIER;
        if(_frontier.vector_engine_part_size > 0)
        {
            copy_if_work = true;
            ParallelPrimitives::copy_if_indexes(in_frontier, _frontier.ids, ve_threshold, _frontier.work_buffer, vertices_count, 0);
        }
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
        {
            copy_if_work = true;
            ParallelPrimitives::copy_if_indexes(in_frontier, &_frontier.ids[_frontier.vector_engine_part_size], vc_threshold - ve_threshold,
                            _frontier.work_buffer, vertices_count, ve_threshold);
        }
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
            copy_if_work = true;
            ParallelPrimitives::copy_if_indexes(in_frontier, &_frontier.ids[_frontier.vector_core_part_size + _frontier.vector_engine_part_size], vertices_count - vc_threshold,
                            _frontier.work_buffer, vertices_count, vc_threshold);
        }
    }
    else
    {
        _frontier.collective_part_type = DENSE_FRONTIER;
    }

    // set type of the whole frontier
    if(_frontier.size == vertices_count)
    {
        _frontier.sparsity_type = ALL_ACTIVE_FRONTIER;
    }
    else if(!copy_if_work) // flags array
    {
        _frontier.sparsity_type = DENSE_FRONTIER;
    }
    else
    {
        _frontier.sparsity_type = SPARSE_FRONTIER;
    }

    tm_copy_if.end();
    tm_wall.end();
    long long work = vertices_count;
    performance_stats.update_gnf_time(tm_wall);
    performance_stats.update_bytes_requested(work*2.0*sizeof(int));
    if(copy_if_work)
        performance_stats.update_bytes_requested(work*2.0*sizeof(int));

    #ifdef __USE_MPI__
    throw "Error: MPI thresholds calculation is not implemented in GraphAbstractionsNEC::generate_new_frontier";
    #endif

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm_flags.print_bandwidth_stats("GNF flags", work, 2.0*sizeof(int));
    if(copy_if_work)
        tm_copy_if.print_bandwidth_stats("GNF copy if", work, 2.0*sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FilterCondition>
void GraphAbstractionsNEC::generate_new_frontier(VGL_Graph &_graph,
                                                 VGL_Frontier &_frontier,
                                                 FilterCondition &&filter_cond)
{
    common_generate_new_frontier(_graph, _frontier, filter_cond, this);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
