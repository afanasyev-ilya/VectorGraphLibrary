#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void SSWP::vgl_dijkstra(VectCSRGraph &_graph,
                        EdgesArray_Vect<_T> &_edges_capacities,
                        VerticesArray<_T> &_widths,
                        int _source_vertex)
{
    VerticesArray<_T> old_widths(_graph, SCATTER);

    _source_vertex = _graph.reorder(_source_vertex, ORIGINAL, SCATTER);

    VGL_GRAPH_ABSTRACTIONS graph_API(_graph);
    VGL_FRONTIER frontier(_graph);

    graph_API.change_traversal_direction(SCATTER, old_widths, _widths, frontier);

    Timer tm;
    tm.start();

    _T inf_val = std::numeric_limits<_T>::max() - MAX_WEIGHT;
    auto init_widths = [_widths, _source_vertex, inf_val] __VGL_COMPUTE_ARGS__
    {
        if(src_id == _source_vertex)
            _widths[_source_vertex] = inf_val;
        else
            _widths[src_id] = 0;
    };
    frontier.set_all_active();
    graph_API.compute(_graph, frontier, init_widths);

    int changes = 0, iterations_count = 0;
    do
    {
        changes = 0;

        auto save_old_widths = [_widths, old_widths] __VGL_COMPUTE_ARGS__
        {
            old_widths[src_id] = _widths[src_id];
        };
        graph_API.compute(_graph, frontier, save_old_widths);

        auto edge_op_push = [_widths, _edges_capacities] __VGL_SCATTER_ARGS__
        {
            _T edge_width = _edges_capacities[global_edge_pos];
            _T new_width = vect_min(_widths[src_id], edge_width);

            if(_widths[dst_id] < new_width)
                _widths[dst_id] = new_width;
        };

        graph_API.scatter(_graph, frontier, edge_op_push);

        auto calculate_changes_count = [_widths, old_widths] __VGL_REDUCE_INT_ARGS__
        {
            int result = 0;
            if(old_widths[src_id] != _widths[src_id])
                result = 1;
            return result;
        };
        changes = graph_API.reduce<int>(_graph, frontier, calculate_changes_count, REDUCE_SUM);

        iterations_count++;
    }
    while(changes);
    tm.end();


    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    performance_stats.print_algorithm_performance_stats("SSWP (Dijkstra, all-active, push)", tm.get_time(), _graph.get_edges_count());
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
#ifdef __USE_NEC_SX_AURORA__
void SSWP::nec_dijkstra(UndirectedCSRGraph &_graph,
                        _TEdgeWeight *_widths,
                        int _source_vertex,
                        AlgorithmTraversalType _traversal_direction)
{
    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    double t1 = omp_get_wtime();
    #endif
    LOAD_UNDIRECTED_CSR_GRAPH_DATA(_graph);
    _TEdgeWeight *old_widths = class_old_widths;

    frontier.set_all_active();

    auto init_widths = [_widths, _source_vertex] __VGL_COMPUTE_ARGS__
    {
        if(src_id == _source_vertex)
            _widths[_source_vertex] = FLT_MAX;
        else
            _widths[src_id] = 0;
    };
    graph_API.compute(_graph, frontier, init_widths);

    int iterations_count = 0;
    int changes = 1;
    while(changes)
    {
        changes = 0;
        float *collective_adjacent_widths = graph_API.get_collective_widths(_graph, frontier);

        auto save_old_widths = [_widths, old_widths] __VGL_COMPUTE_ARGS__
        {
            old_widths[src_id] = _widths[src_id];
        };
        graph_API.compute(_graph, frontier, save_old_widths);

        if(_traversal_direction == PUSH_TRAVERSAL) // PUSH PART
        {
            auto edge_op_push = [adjacent_widths, _widths](int src_id, int dst_id, int local_edge_pos,
                        long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
            {
                float weight = adjacent_widths[global_edge_pos];
                float new_width = vect_min(_widths[src_id], weight);

                if(_widths[dst_id] < new_width)
                    _widths[dst_id] = new_width;
            };

            auto edge_op_collective_push = [collective_adjacent_widths, _widths]
                    (int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
            {
                float weight = collective_adjacent_widths[global_edge_pos];
                float new_width = vect_min(_widths[src_id], weight);

                if(_widths[dst_id] < new_width)
                    _widths[dst_id] = new_width;
            };

            graph_API.advance(_graph, frontier, edge_op_push, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP,
                              edge_op_collective_push, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);
        }
        else if(_traversal_direction == PULL_TRAVERSAL) // PULL PART
        {
            throw "Error: push traversal not supported yet";
        }

        auto calculate_changes_count = [_widths, old_widths] __VGL_COMPUTE_ARGS__->int
        {
            int result = 0;
            if(old_widths[src_id] != _widths[src_id])
                result = 1;
            return result;
        };
        changes = graph_API.reduce<int>(_graph, frontier, calculate_changes_count, REDUCE_SUM);

        iterations_count++;
    }

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    double t2 = omp_get_wtime();
    performance_stats.print_algorithm_performance_stats("all active sswp (dijkstra)", t2 - t1, edges_count, iterations_count);
    #endif
}
#endif*/

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
