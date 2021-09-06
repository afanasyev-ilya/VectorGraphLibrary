#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _T>
double CC::vgl_shiloach_vishkin(VGL_Graph &_graph, VerticesArray<_T> &_components)
{
    VGL_GRAPH_ABSTRACTIONS graph_API(_graph);
    VGL_FRONTIER frontier(_graph);
    frontier.set_all_active();

    _graph.move_to_device();
    _components.move_to_device();
    frontier.move_to_device();

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

        auto edge_op = [_components, hook_changes] __VGL_SCATTER_ARGS__
        {
            int src_val = _components[src_id];
            int dst_val = _components[dst_id];

            if(src_val < dst_val)
            {
                _components[dst_id] = src_val;
                hook_changes[0] = 1;
            }
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
    performance_stats.print_algorithm_performance_stats("CC (Shiloach-Vishkin, GPU)", tm.get_time(), _graph.get_edges_count());
    print_component_sizes(_components);
    #endif

    return performance_stats.get_algorithm_performance(tm.get_time(), _graph.get_edges_count());
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
