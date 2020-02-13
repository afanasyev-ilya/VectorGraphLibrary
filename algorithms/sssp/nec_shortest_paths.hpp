#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _TVertexValue, typename _TEdgeWeight>
void SSSP::nec_dijkstra_partial_active(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                       int _source_vertex,
                                       _TEdgeWeight *_distances)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    double t4 = omp_get_wtime();
    int *was_changes;
    MemoryAPI::allocate_array(&was_changes, vertices_count);
    double t3 = omp_get_wtime();
    cout << "was cahnges init time: " << (t3 - t4) * 1000 << " ms" << endl;

    auto init_distances = [_distances, _source_vertex] (int src_id)
    {
        if(src_id == _source_vertex)
            _distances[_source_vertex] = 0;
        else
            _distances[src_id] = FLT_MAX;
    };

    auto init_changes = [was_changes, _source_vertex] (int src_id)
    {
        if(src_id == _source_vertex)
            was_changes[_source_vertex] = 1;
        else
            was_changes[src_id] = 0;
    };

    auto changes_occurred = [&was_changes] (int src_id)->int
    {
        int res = NEC_NOT_IN_FRONTIER_FLAG;
        if(was_changes[src_id] > 0)
            res = NEC_IN_FRONTIER_FLAG;
        return res;
    };

    graph_API.compute(init_distances, vertices_count);
    graph_API.compute(init_changes, vertices_count);

    frontier.filter(_graph, changes_occurred);

    double compute_time = 0, filter_time = 0;
    int iterations_count = 0;
    while(frontier.size() > 0)
    {
        double t_st = omp_get_wtime();

        float *collective_outgoing_weights = graph_API.get_collective_weights(_graph, frontier);
        auto reset_changes = [was_changes] (int src_id)
        {
            was_changes[src_id] = 0;
        };
        graph_API.compute(reset_changes, vertices_count);

        #pragma omp parallel
        {
            auto edge_op_push = [outgoing_weights, _distances, was_changes]
               (int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos,
                int vector_index, DelayedWriteNEC &delayed_write)
            {
                float weight = outgoing_weights[global_edge_pos];
                float dst_weight = _distances[dst_id];
                float src_weight = _distances[src_id];
                if(dst_weight > src_weight + weight)
                {
                    _distances[dst_id] = src_weight + weight;
                    was_changes[dst_id] = 1;
                    delayed_write.start_write(was_changes, 1, vector_index);
                }
            };

            auto edge_op_collective_push = [collective_outgoing_weights, _distances, was_changes]
                    (int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos,
                            int vector_index, DelayedWriteNEC &delayed_write)
            {
                float weight = collective_outgoing_weights[global_edge_pos];
                float dst_weight = _distances[dst_id];
                float src_weight = _distances[src_id];
                if(dst_weight > src_weight + weight)
                {
                    _distances[dst_id] = src_weight + weight;
                    was_changes[dst_id] = 1;
                    was_changes[src_id] = 1;
                }
            };

            struct VertexPostprocessFunctor
            {
                int *was_changes;
                VertexPostprocessFunctor(int *_was_changes): was_changes(_was_changes) {}
                void operator()(int src_id, int connections_count, DelayedWriteNEC &delayed_write)
                {
                    delayed_write.finish_write_max(was_changes, src_id);
                }
            };
            VertexPostprocessFunctor vertex_postprocess_op(was_changes);

            graph_API.advance(_graph, frontier, edge_op_push, EMPTY_VERTEX_OP, vertex_postprocess_op,
                               edge_op_collective_push, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);
        }
        double t_end = omp_get_wtime();
        compute_time += t_end - t_st;
        double iter_time = t_end - t_st;

        t_st = omp_get_wtime();
        frontier.filter(_graph, changes_occurred);
        t_end = omp_get_wtime();
        filter_time += t_end - t_st;
        iter_time += t_end - t_st;
        cout << endl << "iter " << iterations_count << " perf: " << (edges_count / (iter_time * 1e6)) << " MTEPS" << endl << endl;

        iterations_count++;
    }
    cout << "compute time: " << compute_time*1000 << " ms" << endl;
    cout << "filter time: " << filter_time*1000 << " ms" << endl;
    cout << "inner perf: " << edges_count / (compute_time * 1e6) << " MTEPS" << endl;
    cout << "wall perf: " << edges_count / ((compute_time + filter_time) * 1e6) << " MTEPS" << endl;
    cout << "iterations count: " << iterations_count << endl;
    cout << "perf per iteration: " << iterations_count * (edges_count / (compute_time * 1e6)) << " MTEPS" << endl;

    MemoryAPI::free_array(was_changes);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _TVertexValue, typename _TEdgeWeight>
void SSSP::nec_dijkstra_all_active(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                   int _source_vertex,
                                   _TEdgeWeight *_distances)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    auto init_distances = [_distances, _source_vertex] (int src_id)
    {
        if(src_id == _source_vertex)
            _distances[_source_vertex] = 0;
        else
            _distances[src_id] = FLT_MAX;
    };
    graph_API.compute(init_distances, vertices_count);

    auto all_active = [] (int src_id)->int
    {
        return NEC_IN_FRONTIER_FLAG;
    };
    frontier.filter(_graph, all_active);

    int changes = 1;
    while(changes > 0)
    {
        changes = 0;
        float *collective_outgoing_weights = graph_API.get_collective_weights(_graph, frontier);

        #pragma omp parallel
        {
            NEC_REGISTER_INT(changes, 0);

            auto edge_op_push = [&outgoing_weights, &_distances, &reg_changes](int src_id, int dst_id, int local_edge_pos,
                    long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
            {
                float weight = outgoing_weights[global_edge_pos];
                float dst_weight = _distances[dst_id];
                float src_weight = _distances[src_id];
                if(dst_weight > src_weight + weight)
                {
                    _distances[dst_id] = src_weight + weight;
                    reg_changes[vector_index] = 1;
                }
            };

            auto edge_op_collective_push = [collective_outgoing_weights, _distances, was_changes]
                    (int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos,
                            int vector_index, DelayedWriteNEC &delayed_write)
            {
                float weight = collective_outgoing_weights[global_edge_pos];
                float dst_weight = _distances[dst_id];
                float src_weight = _distances[src_id];
                if(dst_weight > src_weight + weight)
                {
                    _distances[dst_id] = src_weight + weight;
                    reg_changes[vector_index] = 1;
                }
            };

            graph_API.advance((_graph, frontier, edge_op_push, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP,
                               edge_op_collective_push, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);

            changes = register_sum_reduce(reg_changes);
        }
    }
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _TVertexValue, typename _TEdgeWeight>
void SSSP::nec_dijkstra(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                        int _source_vertex,
                        _TEdgeWeight *_distances,
                        AlgorithmFrontierType _frontier_type)
{
    if(_frontier_type == PARTIAL_ACTIVE)
        nec_dijkstra_partial_active(_graph, _source_vertex, _distances);
    else if(_frontier_type == PARTIAL_ACTIVE)
        nec_dijkstra_all_active(_graph, _source_vertex, _distances);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
