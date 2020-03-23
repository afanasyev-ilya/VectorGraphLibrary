#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
FrontierNEC::FrontierNEC(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph)
{
    FrontierNEC(_graph.get_vertices_count());
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FrontierNEC::FrontierNEC(int _vertices_count)
{
    max_frontier_size = _vertices_count;
    MemoryAPI::allocate_array(&frontier_flags, max_frontier_size);
    MemoryAPI::allocate_array(&frontier_ids, max_frontier_size);
    MemoryAPI::allocate_array(&work_buffer, max_frontier_size);

    // by default frontier is all active
    frontier_type = ALL_ACTIVE_FRONTIER;
    current_frontier_size = max_frontier_size;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FrontierNEC::~FrontierNEC()
{
    MemoryAPI::free_array(frontier_flags);
    MemoryAPI::free_array(frontier_ids);
    MemoryAPI::free_array(work_buffer);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight, typename Condition>
void FrontierNEC::filter(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, Condition condition_op)
{
    // fill flags
    if(frontier_type == ALL_ACTIVE_FRONTIER)
    {
        #pragma omp parallel for schedule(static)
        for (int vec_start = 0; vec_start < max_frontier_size; vec_start += VECTOR_LENGTH)
        {
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            #pragma _NEC ivdep
            for (int i = 0; i < VECTOR_LENGTH; i++)
            {
                int src_id = vec_start + i;
                if (src_id < max_frontier_size)
                {
                    frontier_flags[src_id] = condition_op(src_id);
                }
            }
        }
    }
    else if((frontier_type == SPARSE_FRONTIER) || (frontier_type == DENSE_FRONTIER))
    {
        #pragma omp parallel for schedule(static)
        for (int vec_start = 0; vec_start < max_frontier_size; vec_start += VECTOR_LENGTH)
        {
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            #pragma _NEC ivdep
            for (int i = 0; i < VECTOR_LENGTH; i++)
            {
                int src_id = vec_start + i;
                int old_val = frontier_flags[src_id];
                if ((src_id < max_frontier_size) && (old_val == NEC_IN_FRONTIER_FLAG))
                {
                    frontier_flags[src_id] = condition_op(src_id);
                }
            }
        }
    }

    // calculate current frontier size
    int vertices_in_frontier = 0;
    #pragma _NEC ivdep
    #pragma _NEC vector
    #pragma omp parallel for schedule(static) reduction(+: vertices_in_frontier)
    for(int i = 0; i < max_frontier_size; i++)
    {
        vertices_in_frontier += frontier_flags[i];
    }
    current_frontier_size = vertices_in_frontier;

    // chose frontier representation
    if(current_frontier_size == max_frontier_size) // no checks required
    {
        frontier_type = ALL_ACTIVE_FRONTIER;
    }
    else if(double(current_frontier_size)/max_frontier_size > FRONTIER_TYPE_CHANGE_THRESHOLD) // flags array
    {
        frontier_type = DENSE_FRONTIER;
    }
    else // queue + flags for now
    {
        frontier_type = SPARSE_FRONTIER;
        const int ve_threshold = _graph.get_nec_vector_engine_threshold_vertex();
        const int vc_threshold = _graph.get_nec_vector_core_threshold_vertex();
        const int vertices_count = _graph.get_vertices_count();

        vector_engine_part_size = sparse_copy_if(frontier_ids, work_buffer, max_frontier_size, 0, ve_threshold, condition_op);

        vector_core_part_size = sparse_copy_if(&frontier_ids[vector_engine_part_size], work_buffer, max_frontier_size, ve_threshold, vc_threshold, condition_op);

        collective_part_size = sparse_copy_if(&frontier_ids[vector_core_part_size + vector_engine_part_size], work_buffer, max_frontier_size, vc_threshold, vertices_count, condition_op);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierNEC::print_frontier_info()
{
    #pragma omp master
    {
        string status;
        if(frontier_type == ALL_ACTIVE_FRONTIER)
            status = "all active";
        if(frontier_type == SPARSE_FRONTIER)
            status = "sparse";
        if(frontier_type == DENSE_FRONTIER)
            status = "dense";

        cout << "frontier status: " << status << endl;
        cout << "frontier size: " << current_frontier_size << " from " << max_frontier_size << ", " <<
        (100.0 * current_frontier_size) / max_frontier_size << " %" << endl;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierNEC::set_all_active()
{
    frontier_type = ALL_ACTIVE_FRONTIER;
    current_frontier_size = max_frontier_size;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
