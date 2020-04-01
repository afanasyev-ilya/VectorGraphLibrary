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

void FrontierNEC::print_frontier_info()
{
    #pragma omp master
    {
        string status;
        if(type == ALL_ACTIVE_FRONTIER)
            status = "all active";
        if(type == SPARSE_FRONTIER)
            status = "sparse";
        if(type == DENSE_FRONTIER)
            status = "dense";

        cout << "frontier status: " << status << endl;
        cout << "frontier size: " << current_size << " from " << max_size << ", " <<
        (100.0 * current_size) / max_size << " %" << endl;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierNEC::set_all_active()
{
    type = ALL_ACTIVE_FRONTIER;
    current_size = max_size;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
