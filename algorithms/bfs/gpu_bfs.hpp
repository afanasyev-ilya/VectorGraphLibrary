/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _TVertexValue, typename _TEdgeWeight>
void BFS<_TVertexValue, _TEdgeWeight>::allocate_device_result_memory(int _vertices_count, int **_device_levels)
{
    MemoryAPI::allocate_non_managed_array(_device_levels, _vertices_count);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _TVertexValue, typename _TEdgeWeight>
void BFS<_TVertexValue, _TEdgeWeight>::free_device_result_memory(int *_device_levels)
{
    MemoryAPI::free_device_array(_device_levels);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _TVertexValue, typename _TEdgeWeight>
void BFS<_TVertexValue, _TEdgeWeight>::copy_result_to_host(int *_host_levels, int *_device_levels, int _vertices_count)
{
    cudaMemcpy(_host_levels, _device_levels, _vertices_count * sizeof(int), cudaMemcpyDeviceToHost);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _TVertexValue, typename _TEdgeWeight>
void BFS<_TVertexValue, _TEdgeWeight>::gpu_top_down(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                         int *_device_levels,
                                         int _source_vertex)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    _graph.move_to_device();

    int iterations_count = 0;
    double t1 = omp_get_wtime();
    top_down_wrapper<_TVertexValue, _TEdgeWeight>(_graph, _device_levels, _source_vertex, iterations_count);
    double t2 = omp_get_wtime();

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    PerformanceStats::print_performance_stats("BFS (top-down)", t2 - t1, edges_count, iterations_count);
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _TVertexValue, typename _TEdgeWeight>
void BFS<_TVertexValue, _TEdgeWeight>::gpu_direction_optimizing(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                                int *_device_levels,
                                                                int _source_vertex)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    _graph.move_to_device();

    int iterations_count = 0;
    double t1 = omp_get_wtime();
    direction_optimizing_wrapper<_TVertexValue, _TEdgeWeight>(_graph, _device_levels, _source_vertex, iterations_count);
    double t2 = omp_get_wtime();

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    PerformanceStats::print_performance_stats("BFS (direction_optimizing)", t2 - t1, edges_count, iterations_count);
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
