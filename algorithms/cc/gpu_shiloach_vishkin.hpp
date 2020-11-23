#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
void CC::gpu_shiloach_vishkin(UndirectedCSRGraph &_graph,
                              int *_components)
{
    LOAD_UNDIRECTED_CSR_GRAPH_DATA(_graph);

    int *device_components;
    MemoryAPI::allocate_device_array(&device_components, vertices_count);

    int iterations_count = 0;
    double t1 = omp_get_wtime();
    shiloach_vishkin_wrapper(_graph, device_components, iterations_count);
    double t2 = omp_get_wtime();

    MemoryAPI::copy_array_to_host(_components, device_components, vertices_count);
    MemoryAPI::free_array(device_components);

    performance = edges_count / ((t2 - t1)*1e6);

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    PerformanceStats::print_algorithm_performance_stats("shiloach vishkin", t2 - t1, edges_count, iterations_count);
    PerformanceStats::component_stats(_components, vertices_count);
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
