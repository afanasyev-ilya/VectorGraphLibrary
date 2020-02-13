#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _TVertexValue, typename _TEdgeWeight>
void SSSP::gpu_dijkstra(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                        int _source_vertex,
                        _TEdgeWeight *_distances)
{
    //_graph.move_to_device();

    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    _TEdgeWeight *device_distances;
    SAFE_CALL(cudaMalloc((void**)&device_distances, vertices_count * sizeof(_TEdgeWeight)));

    int iterations_count = 0;
    double t1 = omp_get_wtime();
    gpu_dijkstra_wrapper<_TVertexValue, _TEdgeWeight>(_graph, device_distances, _source_vertex, iterations_count);
    double t2 = omp_get_wtime();

    #ifdef __PRINT_DETAILED_STATS__
    print_performance_stats(edges_count, iterations_count, t2 - t1);
    #endif

    SAFE_CALL(cudaMemcpy(_distances, device_distances, vertices_count * sizeof(_TEdgeWeight), cudaMemcpyDeviceToHost));
    SAFE_CALL(cudaFree(device_distances));
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
