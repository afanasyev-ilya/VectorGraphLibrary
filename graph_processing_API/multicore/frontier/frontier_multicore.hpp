#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
FrontierMulticore::FrontierMulticore(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph)
{
    max_size = _graph.get_vertices_count();
    MemoryAPI::allocate_array(&flags, max_size);
    MemoryAPI::allocate_array(&ids, max_size);

    // by default frontier is all active
    type = ALL_ACTIVE_FRONTIER;
    current_size = max_size;

    #pragma omp parallel
    {}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FrontierMulticore::FrontierMulticore(int _vertices_count)
{
    max_size = _vertices_count;
    MemoryAPI::allocate_array(&flags, max_size);
    MemoryAPI::allocate_array(&ids, max_size);

    // by default frontier is all active
    type = ALL_ACTIVE_FRONTIER;
    current_size = max_size;

    #pragma omp parallel
    {}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FrontierMulticore::~FrontierMulticore()
{
    MemoryAPI::free_array(flags);
    MemoryAPI::free_array(ids);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void FrontierMulticore::print_frontier_info(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph)
{
    // TODO
    throw "multicore TODO";
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierMulticore::set_all_active()
{
    type = ALL_ACTIVE_FRONTIER;
    current_size = max_size;

    #pragma omp parallel // dummy for performance evaluation
    {};
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void FrontierMulticore::add_vertex(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int src_id)
{
    //TODO
    throw "multicore TODO";
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void FrontierMulticore::add_vertices(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int *_vertex_ids, int _number_of_vertices)
{
    throw "multicore TODO";
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
