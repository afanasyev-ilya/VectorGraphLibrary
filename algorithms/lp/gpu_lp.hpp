#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__

void LP::gpu_lp(UndirectedCSRGraph &_graph,
                int *_labels,
                GpuActiveConditionType _gpu_active_condition_type,
                int _max_iterations)
{
    LOAD_UNDIRECTED_CSR_GRAPH_DATA(_graph);

    int *device_labels;
    MemoryAPI::allocate_device_array(&device_labels, vertices_count);

    int iterations_count = 0;
    double t1 = omp_get_wtime();
    gpu_lp_wrapper(_graph, device_labels, iterations_count, _gpu_active_condition_type, _max_iterations);
    double t2 = omp_get_wtime();

    MemoryAPI::copy_array_to_host(_labels, device_labels, vertices_count);
    MemoryAPI::free_device_array(device_labels);

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    PerformanceStats::print_algorithm_performance_stats("gpu label propagation", t2 - t1, edges_count, iterations_count);
    #endif

    PerformanceStats::component_stats(_labels, vertices_count);

    if(vertices_count < VISUALISATION_SMALL_GRAPH_VERTEX_THRESHOLD)
    {
        _graph.move_to_host();
        _graph.set_vertex_data_from_array(_labels);
        _graph.save_to_graphviz_file("lp_graph.gv", VISUALISE_AS_DIRECTED);
    }
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
