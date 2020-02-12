#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FrontierNEC::FrontierNEC(int _vertices_count)
{
    max_frontier_size = _vertices_count;
    frontier_flags = new int[max_frontier_size];
    frontier_ids = new int[max_frontier_size];
    work_buffer = new int[max_frontier_size];

    #pragma omp parallel for schedule(static)
    for(int i = 0; i < max_frontier_size; i++)
    {
        frontier_flags[i] = NEC_NOT_IN_FRONTIER_FLAG;
    }

    frontier_type = DENSE_FRONTIER;
    sparse_frontier_size = 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FrontierNEC::~FrontierNEC()
{
    delete []frontier_flags;
    delete []frontier_ids;
    delete []work_buffer;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight, typename Condition>
void FrontierNEC::filter(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, Condition condition_op)
{
    int vertices_in_frontier = 0;

    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma omp parallel for schedule(static) reduction(+: vertices_in_frontier)
    for(int i = 0; i < max_frontier_size; i++)
    {
        if(condition_op(i))
            frontier_flags[i] = NEC_IN_FRONTIER_FLAG;
        else
            frontier_flags[i] = NEC_NOT_IN_FRONTIER_FLAG;

        vertices_in_frontier += frontier_flags[i];
    }

    // if not sparse - set all active
    if(double(vertices_in_frontier)/max_frontier_size > FRONTIER_TYPE_CHANGE_THRESHOLD)
    {
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vector
        #pragma omp parallel for schedule(static)
        for(int i = 0; i < max_frontier_size; i++)
        {
            frontier_flags[i] = NEC_IN_FRONTIER_FLAG;
        }

        frontier_type = DENSE_FRONTIER;
    }
    else
    {
        const int ve_threshold = _graph.get_nec_vector_engine_threshold_vertex();
        const int vc_threshold = _graph.get_nec_vector_core_threshold_vertex();

        sparse_frontier_size = sparse_copy_if(frontier_ids, work_buffer, vc_threshold, max_frontier_size, condition_op);

        frontier_type = SPARSE_FRONTIER;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int FrontierNEC::size()
{
    int size = 0;

    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma omp parallel for schedule(static) reduction(+: size)
    for(int i = 0; i < max_frontier_size; i++)
    {
        size += frontier_flags[i];
    }

    return size;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
