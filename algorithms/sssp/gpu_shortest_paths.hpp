#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _T>
void SSSP::gpu_dijkstra(VectCSRGraph &_graph,
                        EdgesArray<_T> &_weights,
                        VerticesArray<_T> &_distances,
                        int _source_vertex,
                        AlgorithmFrontierType _frontier_type,
                        AlgorithmTraversalType _traversal_direction)
{
    cout << "inside GPU version" << endl;

    //_graph.move_to_device();

    /*LOAD_UNDIRECTED_CSR_GRAPH_DATA(_graph);

    _TEdgeWeight *device_distances;
    MemoryAPI::allocate_non_managed_array(&device_distances, vertices_count);

    int iterations_count = 0;
    double t1 = omp_get_wtime();
    if(_frontier_type == PARTIAL_ACTIVE)
    {
        gpu_dijkstra_partial_active_wrapper(_graph, device_distances, _source_vertex,
                                                                         iterations_count);
    }
    else if(_frontier_type == ALL_ACTIVE)
    {
        gpu_dijkstra_all_active_wrapper(_graph, device_distances, _source_vertex,
                                                                     iterations_count, _traversal_direction);
    }
    double t2 = omp_get_wtime();
    
    performance = edges_count / ((t2 - t1)*1e6);

    MemoryAPI::copy_array_to_host(_distances, device_distances, vertices_count);
    MemoryAPI::free_device_array(device_distances);

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    if(_frontier_type == PARTIAL_ACTIVE)
        PerformanceStats::print_algorithm_performance_stats("SSSP (partial-active, dijkstra)", t2 - t1, edges_count, iterations_count);
    else if(_frontier_type == ALL_ACTIVE)
        PerformanceStats::print_algorithm_performance_stats("SSSP (all-active, dijkstra)", t2 - t1, edges_count, iterations_count);
    #endif*/
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
