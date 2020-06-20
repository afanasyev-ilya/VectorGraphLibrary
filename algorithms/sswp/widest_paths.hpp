#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void WidestPaths<_TVertexValue, _TEdgeWeight>::allocate_result_memory(int _vertices_count, _TEdgeWeight **_widths)
{
    MemoryAPI::allocate_array(_widths, _vertices_count);
    #pragma omp parallel
    {};
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void WidestPaths<_TVertexValue, _TEdgeWeight>::free_result_memory(_TEdgeWeight *_widths)
{
    MemoryAPI::free_array(_widths);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _TVertexValue, typename _TEdgeWeight>
SSWP::WidestPaths(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph):
frontier(_graph.get_vertices_count())
{
    MemoryAPI::allocate_array(&class_old_widths, _graph.get_vertices_count());
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _TVertexValue, typename _TEdgeWeight>
SSWP::~WidestPaths()
{
    MemoryAPI::free_array(class_old_widths);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

