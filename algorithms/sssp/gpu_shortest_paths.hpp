#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _T>
double SSSP::gpu_dijkstra_all_active_push(VGL_Graph &_graph,
                                          EdgesArray<_T> &_weights,
                                          VerticesArray<_T> &_distances,
                                          int _source_vertex)
{
    GraphAbstractionsGPU graph_API(_graph, SCATTER);
    VGL_Frontier frontier(_graph, SCATTER);
    graph_API.change_traversal_direction(SCATTER, _distances, frontier);

    Timer tm;
    tm.start();

    _T inf_val = std::numeric_limits<_T>::max() - MAX_WEIGHT;
    auto init_op = [_distances, _source_vertex, inf_val] __VGL_COMPUTE_ARGS__ {
        if(src_id == _source_vertex)
            _distances[_source_vertex] = 0;
        else
            _distances[src_id] = inf_val;
    };
    frontier.set_all_active();
    graph_API.compute(_graph, frontier, init_op);

    int *changes;
    MemoryAPI::allocate_array(&changes, 1);
    int iterations_count = 0;
    do
    {
        changes[0] = 0;

        auto edge_op = [_weights, _distances, changes] __VGL_SCATTER_ARGS__{
            _T weight = _weights[global_edge_pos];
            _T src_weight = _distances[src_id];
            _T dst_weight = _distances[dst_id];

            if(dst_weight > src_weight + weight)
            {
                _distances[dst_id] = src_weight + weight;
                changes[0] = 1;
            }
        };

        graph_API.scatter(_graph, frontier, edge_op);

        iterations_count++;
    }
    while(changes[0] > 0);

    MemoryAPI::free_array(changes);

    tm.end();

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    performance_stats.print_algorithm_performance_stats("SSSP (Dijkstra, all-active, push, GPU)", tm.get_time(), _graph.get_edges_count());
    #endif

    return performance_stats.get_algorithm_performance(tm.get_time(), _graph.get_edges_count());
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _T>
double SSSP::gpu_dijkstra_all_active_pull(VGL_Graph &_graph,
                                          EdgesArray<_T> &_weights,
                                          VerticesArray<_T> &_distances,
                                          int _source_vertex)
{
    GraphAbstractionsGPU graph_API(_graph, GATHER);
    VGL_Frontier frontier(_graph, GATHER);
    graph_API.change_traversal_direction(GATHER, _distances, frontier);

    Timer tm;
    tm.start();

    _T inf_val = std::numeric_limits<_T>::max() - MAX_WEIGHT;
    auto init_op = [_distances, _source_vertex, inf_val] __VGL_COMPUTE_ARGS__ {
        if(src_id == _source_vertex)
            _distances[_source_vertex] = 0;
        else
            _distances[src_id] = inf_val;
    };
    frontier.set_all_active();
    graph_API.compute(_graph, frontier, init_op);

    int *changes;
    MemoryAPI::allocate_array(&changes, 1);
    int iterations_count = 0;
    do
    {
        changes[0] = 0;

        auto edge_op = [_weights, _distances, changes] __VGL_GATHER_ARGS__ {
            _T weight = _weights[global_edge_pos];
            _T src_weight = __ldg(&_distances[src_id]);
            _T dst_weight = __ldg(&_distances[dst_id]);

            if(src_weight > dst_weight + weight)
            {
                _distances[src_id] = dst_weight + weight;
                changes[0] = 1;
            }
        };

        graph_API.gather(_graph, frontier, edge_op);

        iterations_count++;
    }
    while(changes[0] > 0);

    MemoryAPI::free_array(changes);

    tm.end();

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    performance_stats.print_algorithm_performance_stats("SSSP (Dijkstra, all-active, pull, GPU)", tm.get_time(), _graph.get_edges_count());
    #endif

    return performance_stats.get_algorithm_performance(tm.get_time(), _graph.get_edges_count());
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _T>
double SSSP::gpu_dijkstra_partial_active(VGL_Graph &_graph,
                                         EdgesArray<_T> &_weights,
                                         VerticesArray<_T> &_distances,
                                         int _source_vertex)
{
    GraphAbstractionsGPU graph_API(_graph, SCATTER);
    VGL_Frontier frontier(_graph, SCATTER);
    VerticesArray<char> was_updated(_graph, SCATTER);
    graph_API.change_traversal_direction(SCATTER, _distances, frontier, was_updated);

    Timer tm;
    tm.start();

    _T inf_val = std::numeric_limits<_T>::max() - MAX_WEIGHT;
    auto init_op = [_distances, _source_vertex, inf_val] __VGL_COMPUTE_ARGS__ {
        if(src_id == _source_vertex)
            _distances[_source_vertex] = 0;
        else
            _distances[src_id] = inf_val;
    };
    frontier.set_all_active();
    graph_API.compute(_graph, frontier, init_op);

    auto edge_op = [_weights, _distances, was_updated] __VGL_SCATTER_ARGS__ {
        _T weight = _weights[global_edge_pos];
        _T src_weight = __ldg(&_distances[src_id]);
        _T dst_weight = __ldg(&_distances[dst_id]);

        if(dst_weight > src_weight + weight)
        {
            _distances[dst_id] = src_weight + weight;
            was_updated[dst_id] = 1;
            was_updated[src_id] = 1;
        }
    };

    auto frontier_condition = [was_updated] __VGL_GNF_ARGS__
    {
        if(was_updated[src_id] > 0)
            return IN_FRONTIER_FLAG;
        else
            return NOT_IN_FRONTIER_FLAG;
    };

    frontier.clear();
    frontier.add_vertex(_source_vertex);

    int iterations_count = 0;
    while(frontier.size() > 0)
    {
        cudaMemset(was_updated.get_ptr(), 0, sizeof(char) * _graph.get_vertices_count());
        graph_API.scatter(_graph, frontier, edge_op);
        graph_API.generate_new_frontier(_graph, frontier, frontier_condition);
        iterations_count++;
    }

    tm.end();

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    performance_stats.print_algorithm_performance_stats("SSSP (Dijkstra, partial-active, GPU)", tm.get_time(), _graph.get_edges_count());
    #endif

    return performance_stats.get_algorithm_performance(tm.get_time(), _graph.get_edges_count());
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _T>
double SSSP::vgl_dijkstra(VGL_Graph &_graph,
                          EdgesArray<_T> &_weights,
                          VerticesArray<_T> &_distances,
                          int _source_vertex,
                          AlgorithmFrontierType _frontier_type,
                          AlgorithmTraversalType _traversal_direction)
{
    _graph.move_to_device();
    _weights.move_to_device();
    _distances.move_to_device();

    if(_frontier_type == PARTIAL_ACTIVE)
    {
        return gpu_dijkstra_partial_active(_graph, _weights, _distances, _source_vertex);
    }
    else if(_frontier_type == ALL_ACTIVE)
    {
        if(_traversal_direction == PUSH_TRAVERSAL)
        {
            return gpu_dijkstra_all_active_push(_graph, _weights, _distances, _source_vertex);
        }
        else if(_traversal_direction == PULL_TRAVERSAL)
        {
            return gpu_dijkstra_all_active_pull(_graph, _weights, _distances, _source_vertex);
        }
    }
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
