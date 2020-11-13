#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
void PR::gpu_page_rank(UndirectedCSRGraph &_graph,
                       float *_page_ranks,
                       float _convergence_factor,
                       int _max_iterations,
                       AlgorithmTraversalType _traversal_direction)
{
    LOAD_UNDIRECTED_CSR_GRAPH_DATA(_graph);

    float *device_page_ranks;
    MemoryAPI::allocate_device_array(&device_page_ranks, vertices_count);

    int iterations_count = 0;
    double t1 = omp_get_wtime();
    page_rank_wrapper(_graph, device_page_ranks, _convergence_factor, _max_iterations,
                      iterations_count, _traversal_direction);
    double t2 = omp_get_wtime();

    MemoryAPI::copy_array_to_host(_page_ranks, device_page_ranks, vertices_count);
    MemoryAPI::free_array(device_page_ranks);

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    PerformanceStats::print_algorithm_performance_stats("page rank", t2 - t1, edges_count, iterations_count);
    #endif

    performance_per_iteration = double(iterations_count) * (edges_count/((t2 - t1)*1e6));
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
