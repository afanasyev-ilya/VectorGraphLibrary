#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
FrontierNEC::FrontierNEC(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph)
{
    max_size = _graph.get_vertices_count();
    MemoryAPI::allocate_array(&flags, max_size);
    MemoryAPI::allocate_array(&ids, max_size);
    MemoryAPI::allocate_array(&work_buffer, max_size + VECTOR_LENGTH * MAX_SX_AURORA_THREADS);

    // by default frontier is all active
    type = ALL_ACTIVE_FRONTIER;
    current_size = max_size;

    #pragma omp parallel
    {}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FrontierNEC::FrontierNEC(int _vertices_count)
{
    max_size = _vertices_count;
    MemoryAPI::allocate_array(&flags, max_size);
    MemoryAPI::allocate_array(&ids, max_size);
    MemoryAPI::allocate_array(&work_buffer, max_size);

    // by default frontier is all active
    type = ALL_ACTIVE_FRONTIER;
    current_size = max_size;

    #pragma omp parallel
    {}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FrontierNEC::~FrontierNEC()
{
    MemoryAPI::free_array(flags);
    MemoryAPI::free_array(ids);
    MemoryAPI::free_array(work_buffer);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

string get_status_string(FrontierType _type)
{
    string status;
    if(_type == ALL_ACTIVE_FRONTIER)
        status = "all active";
    if(_type == SPARSE_FRONTIER)
        status = "sparse";
    if(_type == DENSE_FRONTIER)
        status = "dense";

    return status;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void FrontierNEC::print_frontier_info(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph)
{
    #pragma omp master
    {
        const int ve_threshold = _graph.get_nec_vector_engine_threshold_vertex();
        const int vc_threshold = _graph.get_nec_vector_core_threshold_vertex();
        const int vertices_count = _graph.get_vertices_count();

        string status;
        if(type == ALL_ACTIVE_FRONTIER)
            status = "all active";
        if(type == SPARSE_FRONTIER)
            status = "sparse";
        if(type == DENSE_FRONTIER)
            status = "dense";

        cout << "frontier status: " << get_status_string(type) << ":" << get_status_string(vector_engine_part_type) << "/" <<
                 get_status_string(vector_core_part_type) << "/" << get_status_string(collective_part_type) << endl;
        cout << "frontier size: " << current_size << " from " << max_size << ", " <<
        (100.0 * current_size) / max_size << " %" << endl;
        if(ve_threshold > 0)
            cout << 100.0 * vector_engine_part_size / (ve_threshold) << " % active first part" << endl;
        cout << 100.0 * vector_core_part_size / (vc_threshold - ve_threshold) << " % active second part" << endl;
        cout << 100.0 * collective_part_size / (vertices_count - vc_threshold) << " % active third part" << endl;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierNEC::set_all_active()
{
    type = ALL_ACTIVE_FRONTIER;
    current_size = max_size;

    #pragma omp parallel // dummy for performance evaluation
    {};
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void FrontierNEC::add_vertex(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int src_id)
{
    if(current_size > 0)
    {
        throw "VGL ERROR: can not add vertex to non-empty frontier";
    }

    const int ve_threshold = _graph.get_nec_vector_engine_threshold_vertex();
    const int vc_threshold = _graph.get_nec_vector_core_threshold_vertex();
    const int vertices_count = _graph.get_vertices_count();

    ids[0] = src_id;
    flags[src_id] = IN_FRONTIER_FLAG;

    vector_engine_part_type = SPARSE_FRONTIER;
    vector_core_part_type = SPARSE_FRONTIER;
    collective_part_type = SPARSE_FRONTIER;

    vector_engine_part_size = 0;
    vector_core_part_size = 0;
    collective_part_size = 0;

    if(src_id < ve_threshold)
    {
        vector_engine_part_size = 1;
    }
    if((src_id >= ve_threshold) && (src_id < vc_threshold))
    {
        vector_core_part_size = 1;
    }
    if((src_id >= vc_threshold) && (src_id < vertices_count))
    {
        collective_part_size = 1;
    }

    type = SPARSE_FRONTIER;
    current_size = 1;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void FrontierNEC::add_vertices(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int *_vertex_ids, int _number_of_vertices)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    if(current_size > 0)
    {
        throw "VGL ERROR: can not add vertices to non-empty frontier";
    }

    // sort input array
    std::sort(&_vertex_ids[0], &_vertex_ids[_number_of_vertices]);
    //memset(flags, 0, sizeof(int)*max_size);

    // copy ids to frontier inner datastrcuture
    //#pragma _NEC vector
    //#pragma omp parallel for
    for(int idx = 0; idx < _number_of_vertices; idx++)
    {
        ids[idx] = _vertex_ids[idx];
        flags[ids[idx]] = IN_FRONTIER_FLAG;
    }
    current_size = _number_of_vertices;

    //#pragma _NEC vector
    //#pragma omp parallel for
    for(int idx = 0; idx < current_size; idx++)
    {
        const int current_id = ids[idx];
        const int next_id = ids[idx+1];

        int current_size = outgoing_ptrs[current_id + 1] - outgoing_ptrs[current_id];;
        int next_size = 0;
        if(idx < (current_size - 1))
        {
            next_size = outgoing_ptrs[next_id + 1] - outgoing_ptrs[next_id];
        }

        if((current_size > NEC_VECTOR_ENGINE_THRESHOLD_VALUE) && (next_size <= NEC_VECTOR_ENGINE_THRESHOLD_VALUE))
        {
            vector_engine_part_size = idx + 1;
        }

        if((current_size > NEC_VECTOR_CORE_THRESHOLD_VALUE) && (next_size <= NEC_VECTOR_CORE_THRESHOLD_VALUE))
        {
            vector_core_part_size = idx + 1 - vector_engine_part_size;
        }
    }
    collective_part_size = current_size - vector_engine_part_size - vector_core_part_size;

    vector_engine_part_type = SPARSE_FRONTIER;
    vector_core_part_type = SPARSE_FRONTIER;
    collective_part_type = SPARSE_FRONTIER;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
