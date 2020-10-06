#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__

void CC::nec_shiloach_vishkin(ExtendedCSRGraph &_graph,
                              int *_components)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    frontier.set_all_active();
    auto init_components_op = [&_components] (int src_id, int connections_count, int vector_index)
    {
        _components[src_id] = src_id;
    };
    graph_API.compute(_graph, frontier, init_components_op);

    double t1 = omp_get_wtime();
    int hook_changes = 1, jump_changes = 1;
    int iteration = 0;
    while(hook_changes)
    {
        #pragma omp parallel
        {
            hook_changes = 0;
            NEC_REGISTER_INT(hook_changes, 0);

            auto edge_op = [_components, &reg_hook_changes](int src_id, int dst_id, int local_edge_pos,
                    long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
            {
                int src_val = _components[src_id];
                int dst_val = _components[dst_id];

                if(src_val < dst_val)
                {
                    _components[dst_id] = src_val;
                    reg_hook_changes[vector_index] = 1;
                }

                if(src_val > dst_val)
                {
                    delayed_write.start_write(_components, dst_val, vector_index);
                    reg_hook_changes[vector_index] = 1;
                }
            };

            auto edge_op_collective = [_components, &reg_hook_changes](int src_id, int dst_id, int local_edge_pos,
                    long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
            {
                int src_val = _components[src_id];
                int dst_val = _components[dst_id];

                if(src_val < dst_val)
                {
                    _components[dst_id] = src_val;
                    reg_hook_changes[vector_index] = 1;
                }

                if(src_val > dst_val)
                {
                    _components[src_id] = dst_val;
                    reg_hook_changes[vector_index] = 1;
                }
            };

            auto vertex_preprocess_op = [_components]
                        (int src_id, int connections_count, int vector_index, DelayedWriteNEC &delayed_write)
            {
                delayed_write.init(_components, _components[src_id]);
            };

            auto vertex_postprocess_op = [_components]
                    (int src_id, int connections_count, int vector_index, DelayedWriteNEC &delayed_write)
            {
                delayed_write.finish_write_min(_components, src_id);
            };

            graph_API.advance(_graph, frontier, edge_op, vertex_preprocess_op, vertex_postprocess_op, edge_op_collective, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);

            #pragma omp atomic
            hook_changes += register_sum_reduce(reg_hook_changes);
        }

        jump_changes = 1;
        while(jump_changes)
        {
            jump_changes = 0;
            NEC_REGISTER_INT(jump_changes, 0);

            auto jump_op = [_components, &reg_jump_changes](int src_id, int connections_count, int vector_index)
            {
                int src_val = _components[src_id];
                int src_src_val = _components[src_val];

                if(src_val != src_src_val)
                {
                    _components[src_id] = src_src_val;
                    reg_jump_changes[vector_index]++;
                }
            };

            graph_API.compute(_graph, frontier, jump_op);

            jump_changes += register_sum_reduce(reg_jump_changes);
        }

        iteration++;
    }
    double t2 = omp_get_wtime();

    performance = edges_count / ((t2 - t1)*1e6);

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    PerformanceStats::print_performance_stats("shiloach vishkin", t2 - t1, edges_count, iteration);
    PerformanceStats::component_stats(_components, vertices_count);
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
