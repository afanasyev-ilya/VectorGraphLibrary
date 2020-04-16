#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _TVertexValue, typename _TEdgeWeight>
void CC::nec_shiloach_vishkin(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                              int *_components)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    double t1 = omp_get_wtime();
    frontier.set_all_active();
    auto init_components_op = [&_components] (int src_id, int connections_count, int vector_index)
    {
        _components[src_id] = src_id;
    };
    graph_API.compute(_graph, frontier, init_components_op);

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

                int dst_dst_val = -1;
                if(src_val < dst_val)
                    dst_dst_val = _components[dst_val];

                if((src_val < dst_val) && (dst_val == dst_dst_val))
                {
                    _components[dst_val] = src_val;
                    reg_hook_changes[vector_index] = 1;
                }
            };

            graph_API.advance(_graph, frontier, edge_op);

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

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    PerformanceStats::print_performance_stats("shiloach vishkin", t2 - t1, edges_count, iteration);
    PerformanceStats::component_stats(_components, vertices_count);
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
