#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _TVertexValue, typename _TEdgeWeight>
void SSSP::gpu_dijkstra(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                        _TEdgeWeight *_distances,
                        int _source_vertex)
{
    //_graph.move_to_device();

    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    _TEdgeWeight *device_distances;
    MemoryAPI::allocate_device_array(&device_distances, vertices_count);

    int iterations_count = 0;
    double t1 = omp_get_wtime();
    gpu_dijkstra_wrapper<_TVertexValue, _TEdgeWeight>(_graph, device_distances, _source_vertex, iterations_count);
    double t2 = omp_get_wtime();

    MemoryAPI::copy_array_to_host(_distances, device_distances, vertices_count);
    MemoryAPI::free_device_array(device_distances);

    #ifdef __PRINT_DETAILED_STATS__
    performance_stats(edges_count, iterations_count, t2 - t1);
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
