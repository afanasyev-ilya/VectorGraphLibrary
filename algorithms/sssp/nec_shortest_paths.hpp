#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _TVertexValue, typename _TEdgeWeight>
void SSSP::nec_dijkstra_partial_active(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                       _TEdgeWeight *_distances,
                                       int _source_vertex)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    int *was_changes;
    MemoryAPI::allocate_array(&was_changes, vertices_count);

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

    double t1 = omp_get_wtime();
    int iterations_count = 0;
    while(frontier.size() > 0)
    {
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

        frontier.filter(_graph, changes_occurred);
        iterations_count++;
    }
    double t2 = omp_get_wtime();

    performance_stats("partial active sssp (dijkstra)", t2 - t1, edges_count, iterations_count);

    MemoryAPI::free_array(was_changes);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _TVertexValue, typename _TEdgeWeight>
void SSSP::nec_dijkstra_all_active(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                   _TEdgeWeight *_distances,
                                   int _source_vertex,
                                   TraversalDirection _traversal_direction)
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

    double t1 = omp_get_wtime();
    int iterations_count = 0;
    int changes = 1;
    while(changes > 0)
    {
        changes = 0;
        float *collective_outgoing_weights = graph_API.get_collective_weights(_graph, frontier);

        #pragma omp parallel
        {
            NEC_REGISTER_INT(changes, 0);

            auto edge_op_push = [outgoing_weights, _distances, &reg_changes](int src_id, int dst_id, int local_edge_pos,
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

            auto edge_op_collective_push = [collective_outgoing_weights, _distances, &reg_changes]
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

            float shortest_path = 0;

            auto edge_op_pull = [outgoing_weights, _distances, &reg_changes](int src_id, int dst_id, int local_edge_pos,
                    long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
            {
                float weight = outgoing_weights[global_edge_pos];
                float dst_weight = _distances[dst_id];
                if(delayed_write.flt_vec_reg[vector_index] > dst_weight + weight)
                {
                    delayed_write.flt_vec_reg[vector_index] = dst_weight + weight;
                    //delayed_write.start_write(_distances, dst_weight + weight, vector_index);
                    reg_changes[vector_index] = 1;
                }
            };

            struct VertexPreprocessFunctor
            {
                float *_distances;
                VertexPreprocessFunctor(float *distances): _distances(distances) {}
                void operator()(int src_id, int connections_count, DelayedWriteNEC &delayed_write)
                {
                    delayed_write.init(_distances, _distances[src_id]);
                }
            };
            VertexPreprocessFunctor vertex_preprocess_op(_distances);

            struct VertexPostprocessFunctor
            {
                float *_distances;
                VertexPostprocessFunctor(float *distances): _distances(distances) {}
                void operator()(int src_id, int connections_count, DelayedWriteNEC &delayed_write)
                {
                    delayed_write.finish_write_min(_distances, src_id);
                }
            };
            VertexPostprocessFunctor vertex_postprocess_op(_distances);

            auto edge_op_collective_pull = [collective_outgoing_weights, _distances, &reg_changes]
                    (int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos,
                            int vector_index, DelayedWriteNEC &delayed_write)
            {
                float weight = collective_outgoing_weights[global_edge_pos];
                float dst_weight = _distances[dst_id];
                float src_weight = _distances[src_id];
                if(src_weight > dst_weight + weight)
                {
                    _distances[src_id] = dst_weight + weight;
                    reg_changes[vector_index] = 1;
                }
            };

            if(_traversal_direction == PUSH_TRAVERSAL)
                graph_API.advance(_graph, frontier, edge_op_push, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP,
                                   edge_op_collective_push, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);
            else if(_traversal_direction == PULL_TRAVERSAL)
                graph_API.advance(_graph, frontier, edge_op_pull, vertex_preprocess_op, vertex_postprocess_op,
                                   edge_op_collective_pull, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);

            changes = register_sum_reduce(reg_changes);
        }
        iterations_count++;
    }
    double t2 = omp_get_wtime();

    performance_stats("all active sssp (dijkstra)", t2 - t1, edges_count, iterations_count);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _TVertexValue, typename _TEdgeWeight>
void SSSP::nec_dijkstra(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                        _TEdgeWeight *_distances,
                        int _source_vertex,
                        AlgorithmFrontierType _frontier_type,
                        TraversalDirection _traversal_direction)
{
    if(_frontier_type == PARTIAL_ACTIVE)
        nec_dijkstra_partial_active(_graph, _distances, _source_vertex);
    else if(_frontier_type == ALL_ACTIVE)
        nec_dijkstra_all_active(_graph, _distances, _source_vertex, _traversal_direction);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
