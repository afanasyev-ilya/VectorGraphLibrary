#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _T>
void CC::nec_shiloach_vishkin(VectCSRGraph &_graph,
                              VerticesArray<_T> &_components)
{
    GraphAbstractionsNEC graph_API(_graph);
    FrontierNEC frontier(_graph);
    graph_API.change_traversal_direction(SCATTER, _components, frontier);

    #pragma omp parallel
    {};

    Timer tm;
    tm.start();

    frontier.set_all_active();
    auto init_components_op = [&_components] (int src_id, int connections_count, int vector_index)
    {
        _components[src_id] = src_id;
    };
    graph_API.compute(_graph, frontier, init_components_op);

    int hook_changes = 1, jump_changes = 1;
    int iterations_count = 0;
    while(hook_changes)
    {
        #pragma omp parallel
        {
            hook_changes = 0;
            NEC_REGISTER_INT(hook_changes, 0);

            int *components_ptr = _components.get_ptr();

            auto edge_op = [&components_ptr, &reg_hook_changes](int src_id, int dst_id, int local_edge_pos,
                long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
            {
                int src_val = components_ptr[src_id];
                int dst_val = components_ptr[dst_id];

                if(src_val < dst_val)
                {
                    components_ptr[dst_id] = src_val;
                    reg_hook_changes[vector_index] = 1;
                }

                if(src_val > dst_val)
                {
                    delayed_write.start_write(components_ptr, dst_val, vector_index);
                    reg_hook_changes[vector_index] = 1;
                }
            };

            auto edge_op_collective = [components_ptr, &reg_hook_changes](int src_id, int dst_id, int local_edge_pos,
                long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
            {
                int src_val = components_ptr[src_id];
                int dst_val = components_ptr[dst_id];

                if(src_val < dst_val)
                {
                    components_ptr[dst_id] = src_val;
                    reg_hook_changes[vector_index] = 1;
                }

                if(src_val > dst_val)
                {
                    components_ptr[src_id] = dst_val;
                    reg_hook_changes[vector_index] = 1;
                }
            };

            auto vertex_preprocess_op = [components_ptr]
                        (int src_id, int connections_count, int vector_index, DelayedWriteNEC &delayed_write)
            {
                delayed_write.init(components_ptr, components_ptr[src_id]);
            };

            auto vertex_postprocess_op = [components_ptr]
                    (int src_id, int connections_count, int vector_index, DelayedWriteNEC &delayed_write)
            {
                delayed_write.finish_write_min(components_ptr, src_id);
            };

            graph_API.scatter(_graph, frontier, edge_op, vertex_preprocess_op, vertex_postprocess_op, edge_op_collective, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);

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

        iterations_count++;
    }
    tm.end();

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    PerformanceStats::print_algorithm_performance_stats("CC (Shiloach-Vishkin, NEC)", tm.get_time(),
                                                        _graph.get_edges_count(), iterations_count);
    print_component_sizes(_components);
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
