#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _TVertexValue, typename _TEdgeWeight>
SSSP::ShortestPaths(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph):
frontier(_graph.get_vertices_count())
{
    MemoryAPI::allocate_array(&class_old_distances, _graph.get_vertices_count());
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _TVertexValue, typename _TEdgeWeight>
SSSP::ShortestPaths(EdgesListGraph<_TVertexValue, _TEdgeWeight> &_graph):
frontier(_graph.get_vertices_count())
{
    MemoryAPI::allocate_array(&class_old_distances, _graph.get_vertices_count());
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _TVertexValue, typename _TEdgeWeight>
SSSP::ShortestPaths(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph)
{

}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _TVertexValue, typename _TEdgeWeight>
SSSP::~ShortestPaths()
{
    MemoryAPI::free_array(class_old_distances);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _TVertexValue, typename _TEdgeWeight>
SSSP::~ShortestPaths()
{

}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void SSSP::allocate_result_memory(int _vertices_count, _TEdgeWeight **_distances)
{
    MemoryAPI::allocate_array(_distances, _vertices_count);
    #pragma omp parallel
    {};
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void SSSP::free_result_memory(_TEdgeWeight *_distances)
{
    MemoryAPI::free_array(_distances);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void SSSP::reorder_result(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                          _TEdgeWeight *_distances)
{
    int vertices_count = _graph.get_vertices_count();
    int *reordered_ids = _graph.get_reordered_vertex_ids();

    int *tmp_distances;
    MemoryAPI::allocate_array(&tmp_distances, vertices_count);

    for(int i = 0; i < vertices_count; i++)
    {
        tmp_distances[i] = _distances[reordered_ids[i]];
    }

    for(int i = 0; i < vertices_count; i++)
    {
        _distances[i] = tmp_distances[i];
    }

    MemoryAPI::free_array(tmp_distances);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

