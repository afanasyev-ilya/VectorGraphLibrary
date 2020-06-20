#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _TVertexValue, typename _TEdgeWeight>
void SSWP::nec_dijkstra(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                        _TEdgeWeight *_widths,
                        int _source_vertex,
                        TraversalDirection _traversal_direction)
{
    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    double t1 = omp_get_wtime();
    #endif
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);
    _TEdgeWeight *old_widths = class_old_widths;

    frontier.set_all_active();

    auto init_widths = [_widths, _source_vertex] (int src_id, int connections_count, int vector_index)
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
        float *collective_outgoing_weights = graph_API.get_collective_weights(_graph, frontier);

        auto save_old_widths = [_widths, old_widths] (int src_id, int connections_count, int vector_index)
        {
            old_widths[src_id] = _widths[src_id];
        };
        graph_API.compute(_graph, frontier, save_old_widths);

        if(_traversal_direction == PUSH_TRAVERSAL) // PUSH PART
        {
            auto edge_op_push = [outgoing_weights, _widths](int src_id, int dst_id, int local_edge_pos,
                        long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
            {
                float weight = outgoing_weights[global_edge_pos];
                float new_width = vect_min(_widths[src_id], weight);

                if(_widths[dst_id] < new_width)
                    _widths[dst_id] = new_width;
            };

            auto edge_op_collective_push = [collective_outgoing_weights, _widths]
                    (int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
            {
                float weight = collective_outgoing_weights[global_edge_pos];
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

        auto calculate_changes_count = [_widths, old_widths] (int src_id, int connections_count, int vector_index)->int
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
    PerformanceStats::print_performance_stats("all active sswp (dijkstra)", t2 - t1, edges_count, iterations_count);
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
