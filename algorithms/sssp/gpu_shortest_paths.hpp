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
    Timer tm;
    _graph.move_to_device();
    _weights.move_to_device();
    _distances.move_to_device();

    int iterations_count = 0;
    tm.start();
    if(_frontier_type == PARTIAL_ACTIVE)
    {
        _source_vertex = _graph.reorder(_source_vertex, ORIGINAL, SCATTER);
        gpu_dijkstra_partial_active_wrapper(_graph, _weights, _distances, _source_vertex,
                                            iterations_count);
    }
    else if(_frontier_type == ALL_ACTIVE)
    {
        if(_traversal_direction == PUSH_TRAVERSAL)
        {
            _source_vertex = _graph.reorder(_source_vertex, ORIGINAL, SCATTER);
            gpu_dijkstra_all_active_push_wrapper(_graph, _weights, _distances, _source_vertex,
                                                 iterations_count);
        }
        else if(_traversal_direction == PULL_TRAVERSAL)
        {
            _source_vertex = _graph.reorder(_source_vertex, ORIGINAL, GATHER);
            gpu_dijkstra_all_active_pull_wrapper(_graph, _weights, _distances, _source_vertex,
                                                 iterations_count);
        }

    }
    tm.end();

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    if(_frontier_type == PARTIAL_ACTIVE)
        PerformanceStats::print_algorithm_performance_stats("SSSP (partial-active, dijkstra)", tm.get_time(), _graph.get_edges_count(), iterations_count);
    else if(_frontier_type == ALL_ACTIVE)
        PerformanceStats::print_algorithm_performance_stats("SSSP (all-active, dijkstra)", tm.get_time(), _graph.get_edges_count(), iterations_count);
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
