#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _TVertexValue, typename _TEdgeWeight>
void CC::gpu_shiloach_vishkin(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                              int *_components)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    int *device_components;
    MemoryAPI::allocate_device_array(&device_components, vertices_count);

    int iterations_count = 0;
    double t1 = omp_get_wtime();
    shiloach_vishkin_wrapper<_TVertexValue, _TEdgeWeight>(_graph, device_components, iterations_count);
    double t2 = omp_get_wtime();

    MemoryAPI::copy_array_to_host(_components, device_components, vertices_count);
    MemoryAPI::free_device_array(device_components);

    //#ifdef __PRINT_DETAILED_STATS__
    performance_stats("shiloach vishkin", t2 - t1, edges_count, iterations_count);
    component_stats(_components, vertices_count);
    //#endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
