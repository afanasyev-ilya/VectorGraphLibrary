#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _TVertexValue, typename _TEdgeWeight>
void PR::gpu_page_rank(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                       float *_page_ranks,
                       float _convergence_factor,
                       int _max_iterations)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    float *device_page_ranks;
    MemoryAPI::allocate_device_array(&device_page_ranks, vertices_count);

    int iterations_count = 0;
    double t1 = omp_get_wtime();
    page_rank_wrapper(_graph, device_page_ranks, _convergence_factor, _max_iterations,
                      iterations_count);
    double t2 = omp_get_wtime();

    MemoryAPI::copy_array_to_host(_page_ranks, device_page_ranks, vertices_count);
    MemoryAPI::free_device_array(device_page_ranks);

    #ifdef __PRINT_DETAILED_STATS__
    PerformanceStats::print_performance_stats("page rank", t2 - t1, edges_count, iterations_count);
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
