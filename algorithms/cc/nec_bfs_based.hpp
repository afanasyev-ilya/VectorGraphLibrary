#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
template <typename _T>
int CC::select_pivot(VectCSRGraph &_graph,
                     VGL_GRAPH_ABSTRACTIONS &_graph_API,
                     VGL_FRONTIER &_frontier,
                     VerticesArray<_T> &_components)
{
    NEC_REGISTER_INT(pivots, _graph.get_vertices_count() + 1);

    auto select_pivot = [&_components, &reg_pivots] __VGL_COMPUTE_ARGS__
    {
        if(_components[src_id] == COMPONENT_UNSET)
            reg_pivots[vector_index] = src_id;
    };
    _frontier.set_all_active();
    _graph_API.compute(_graph, _frontier, select_pivot);

    int pivot = _graph.get_vertices_count() + 1;
    #pragma _NEC vector
    for(int i = 0; i < VECTOR_LENGTH; i++)
    {
        if(reg_pivots[i] < pivot)
            pivot = reg_pivots[i];
    }

    if(pivot >= _graph.get_vertices_count())
        pivot = -1;

    return pivot;
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
template <typename _T>
void CC::nec_bfs_based(VectCSRGraph &_graph, VerticesArray<_T> &_components)
{
    VGL_GRAPH_ABSTRACTIONS graph_API(_graph, SCATTER);
    VGL_FRONTIER frontier(_graph, SCATTER);

    VerticesArray<_T> bfs_levels(_graph, SCATTER);

    #pragma omp parallel
    {};

    Timer tm;
    tm.start();

    //
    VectorCSRGraph *outgoing_graph_ptr = _graph.get_outgoing_data();
    LOAD_VECTOR_CSR_GRAPH_DATA((*outgoing_graph_ptr));

    //int vertices_count = _graph.get_vertices_count();

    frontier.set_all_active();
    auto init_and_remove_zero_nodes = [&_components, vertices_count] __VGL_COMPUTE_ARGS__
    {
        if(connections_count == 0)
            _components[src_id] = src_id + vertices_count;
        else
            _components[src_id] = COMPONENT_UNSET;
    };
    graph_API.compute(_graph, frontier, init_and_remove_zero_nodes);

    int current_component_num = FIRST_COMPONENT;
    do
    {
        int pivot = select_pivot(_graph, graph_API, frontier, _components);
        cout << "pivot: " << pivot << " " << _components[pivot] << " " << vertex_pointers[pivot + 1] - vertex_pointers[pivot] << endl;

        if(pivot == -1)
        {
            break;
        }

        frontier.clear();
        frontier.add_vertex(pivot);
        _components[pivot] = COMPONENT_FIRST_BFS_LEVEL;

        int current_level = COMPONENT_FIRST_BFS_LEVEL;
        while(frontier.size() > 0)
        {
            auto edge_op = [&_components, current_level] __VGL_SCATTER_ARGS__
            {
                int dst_comp = _components[dst_id];
                if(dst_comp == COMPONENT_UNSET)
                {
                    _components[dst_id] = current_level - 1;
                }
            };

            graph_API.scatter(_graph, frontier, edge_op);

            auto on_next_level = [&_components, current_level] __VGL_GNF_ARGS__
            {
                int result = NOT_IN_FRONTIER_FLAG;
                if(_components[src_id] == (current_level - 1))
                    result = IN_FRONTIER_FLAG;
                return result;
            };

            graph_API.generate_new_frontier(_graph, frontier, on_next_level);

            current_level--;
        }

        frontier.set_all_active();
        auto mark_component = [&_components, current_component_num] __VGL_COMPUTE_ARGS__
        {
            if(_components[src_id] <= COMPONENT_FIRST_BFS_LEVEL)
                _components[src_id] = current_component_num;
        };
        graph_API.compute(_graph, frontier, mark_component);
        current_component_num++;
    } while (true);
    tm.end();

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    performance_stats.print_algorithm_performance_stats("CC (BFS-based, NEC)", tm.get_time(), _graph.get_edges_count());
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

