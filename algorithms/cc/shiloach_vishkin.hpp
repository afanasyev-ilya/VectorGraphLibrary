#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _TVertexValue, typename _TEdgeWeight>
void ConnectedComponents<_TVertexValue,_TEdgeWeight>::nec_shiloach_vishkin(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                                           int *_components)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    GraphPrimitivesNEC graph_API;
    FrontierNEC frontier(_graph.get_vertices_count());

    double t1 = omp_get_wtime();

    auto init_components_op = [_components] (int src_id)
    {
        _components[src_id] = src_id;
    };
    graph_API.compute(init_components_op, vertices_count);

    auto all_active = [] (int src_id)->int
    {
        return NEC_IN_FRONTIER_FLAG;
    };
    frontier.filter(_graph, all_active);

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

        t1 = omp_get_wtime();
        jump_changes = 1;
        while(jump_changes)
        {
            jump_changes = 0;
            NEC_REGISTER_INT(jump_changes, 0);

            auto jump_op = [_components, &reg_jump_changes](int src_id)
            {
                int src_val = _components[src_id];
                int src_src_val = _components[src_val];
                int vector_index = src_id % VECTOR_LENGTH;

                if(src_val != src_src_val)
                {
                    _components[src_id] = src_src_val;
                    reg_jump_changes[vector_index]++;
                }
            };

            graph_API.compute(jump_op, vertices_count);

            jump_changes += register_sum_reduce(reg_jump_changes);
        }

        iteration++;
    }
    double t2 = omp_get_wtime();

    component_stats(_components, vertices_count);
    performance_stats("shiloach vishkin", t2 - t1, edges_count, iteration);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
