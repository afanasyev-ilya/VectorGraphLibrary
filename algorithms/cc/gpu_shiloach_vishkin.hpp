#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _T>
void CC::gpu_shiloach_vishkin(VectCSRGraph &_graph, VerticesArray<_T> &_components)
{
    GraphAbstractionsGPU graph_API(_graph);
    FrontierGPU frontier(_graph);
    graph_API.change_traversal_direction(SCATTER, _components, frontier);

    frontier.set_all_active();
    auto init_components_op = [_components] __VGL_COMPUTE_ARGS__
    {
        _components[src_id] = src_id;
    };
    graph_API.compute(_graph, frontier, init_components_op);

    Timer tm;
    tm.start();

    int *hook_changes, *jump_changes;
    cudaMallocManaged(&hook_changes, sizeof(int));
    cudaMallocManaged(&jump_changes, sizeof(int));

    int iterations_count = 0;
    do
    {
        hook_changes[0] = 0;

        auto edge_op = [_components, hook_changes] __device__(int src_id, int dst_id, int local_edge_pos,
                long long int global_edge_pos, int vector_index)
        {
            int src_val = _components[src_id];
            int dst_val = _components[dst_id];

            if(src_val < dst_val)
            {
                _components[dst_id] = src_val;
                hook_changes[0] = 1;
            }

            /*if(src_val > dst_val)
            {
                _components[src_id] = dst_val;
                hook_changes[0] = 1;
            }*/
        };

        graph_API.scatter(_graph, frontier, edge_op);

        do
        {
            jump_changes[0] = 0;
            auto jump_op = [_components, jump_changes] __VGL_COMPUTE_ARGS__
            {
                int src_label = _components[src_id];
                int parent_label = _components[src_label];

                if(src_label != parent_label)
                {
                    _components[src_id] = parent_label;
                    jump_changes[0] = 0;
                }
            };

            graph_API.compute(_graph, frontier, jump_op);
        } while(jump_changes[0] > 0);

        iterations_count++;
    } while(hook_changes[0] > 0);
    tm.end();

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    PerformanceStats::print_algorithm_performance_stats("CC (Shiloach-Vishkin, GPU)", tm.get_time(),
                                                        _graph.get_edges_count(), iterations_count);
    print_component_sizes(_components);
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
