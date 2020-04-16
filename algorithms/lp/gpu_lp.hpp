#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _TVertexValue, typename _TEdgeWeight>
void LP::gpu_lp(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int *_labels)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    int *device_labels;
    MemoryAPI::allocate_device_array(&device_labels, vertices_count);

    int iterations_count = 0;
    double t1 = omp_get_wtime();
    gpu_lp_wrapper(_graph, device_labels, iterations_count);
    double t2 = omp_get_wtime();

    MemoryAPI::copy_array_to_host(_labels, device_labels, vertices_count);
    MemoryAPI::free_device_array(device_labels);

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    PerformanceStats::print_performance_stats("gpu label propagation", t2 - t1, edges_count, iterations_count);
    PerformanceStats::component_stats(_labels, vertices_count);
    #endif

    if(vertices_count < 30)
    {
        _graph.move_to_host();
        _graph.set_vertex_data_from_array(_labels);
        _graph.save_to_graphviz_file("lp_graph.gv", VISUALISE_AS_DIRECTED);
    }
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
