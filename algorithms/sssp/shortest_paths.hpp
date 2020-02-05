#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ShortestPaths<_TVertexValue, _TEdgeWeight>::allocate_result_memory(int _vertices_count, _TEdgeWeight **_distances)
{
    *_distances = new _TEdgeWeight[_vertices_count];
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ShortestPaths<_TVertexValue, _TEdgeWeight>::free_result_memory(_TEdgeWeight *_distances)
{
    delete[] _distances;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ShortestPaths<_TVertexValue, _TEdgeWeight>::reorder_result(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                                _TEdgeWeight *_distances)
{
    int vertices_count = _graph.get_vertices_count();
    int *reordered_ids = _graph.get_reordered_vertex_ids();

    _TEdgeWeight *tmp_distances = new _TEdgeWeight[vertices_count];

    for(int i = 0; i < vertices_count; i++)
    {
        tmp_distances[i] = _distances[reordered_ids[i]];
    }

    for(int i = 0; i < vertices_count; i++)
    {
        _distances[i] = tmp_distances[i];
    }

    delete []tmp_distances;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ShortestPaths<_TVertexValue, _TEdgeWeight>::reorder_result(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                                _TEdgeWeight *_distances)
{
    int vertices_count = _graph.get_vertices_count();
    int *reordered_ids = _graph.get_reordered_vertex_ids();

    _TEdgeWeight *tmp_distances = new _TEdgeWeight[vertices_count];

    for(int i = 0; i < vertices_count; i++)
    {
        tmp_distances[i] = _distances[reordered_ids[i]];
    }

    for(int i = 0; i < vertices_count; i++)
    {
        _distances[i] = tmp_distances[i];
    }

    delete []tmp_distances;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ShortestPaths<_TVertexValue, _TEdgeWeight>::print_performance_stats(long long _edges_count,
                                                                         int _iterations_count,
                                                                         double _wall_time)
{
    int bytes_per_edge = sizeof(int) + 2*sizeof(_TEdgeWeight);
    cout << "Time               : " << _wall_time << endl;
    cout << "Performance        : " << ((double)_edges_count) / (_wall_time * 1e6) << " MFLOPS" << endl;
    cout << "Iteration    count : " << _iterations_count << endl;
    cout << "Perf. per iteration: " << _iterations_count * ((double)_edges_count) / (_wall_time * 1e6) << " MFLOPS" << endl;
    cout << "Bandwidth          : " << _iterations_count*((double)_edges_count * (bytes_per_edge)) / (_wall_time * 1e9) << " GB/s" << endl;
    cout << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ShortestPaths<_TVertexValue, _TEdgeWeight>::lib_dijkstra(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                              int _source_vertex,
                                                              _TEdgeWeight *_distances)
{
    double t1, t2;
    t1 = omp_get_wtime();
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);
    GraphPrimitivesNEC operations;
    FrontierNEC frontier(vertices_count);
    t2 = omp_get_wtime();
    cout << "alloc time: " << (t2 - t1)*1000 << " ms" << endl;

    int *was_changes = new int[vertices_count];
    int *new_was_changes = new int[vertices_count];
    #pragma omp parallel for
    for(int i = 0; i < vertices_count; i++)
    {
        was_changes[i] = 0;
    }
    was_changes[_source_vertex] = 1;

    t1 = omp_get_wtime();
    //frontier.set_all_active();

    auto init_op = [_distances, _source_vertex] (int src_id)
    {
        if(src_id == _source_vertex)
            _distances[_source_vertex] = 0;
        else
            _distances[src_id] = FLT_MAX;
    };

    auto frontier_condition = [&was_changes] (int idx)->bool{
        if(was_changes[idx] > 0)
            return true;
        else
            return false;
    };

    #pragma omp parallel
    {
        operations.init(vertices_count, init_op);
    }
    //frontier.generate_frontier(_graph, frontier_condition);
    frontier.set_all_active();
    //frontier.set_frontier_flags(frontier_condition);

    t2 = omp_get_wtime();
    cout << "init time: " << (t2 - t1)*1000 << " ms" << endl;

    t1 = omp_get_wtime();
    int changes = 1;
    double compute_time = 0;
    while(changes)
    {
        #pragma omp parallel for
        for(int i = 0; i < vertices_count; i++)
        {
            new_was_changes[i] = 0;
        }

        changes = 0;
        double t_st = omp_get_wtime();
        #pragma omp parallel
        {
            NEC_REGISTER_INT(changes, 0);

            auto edge_op = [&outgoing_weights, &_distances, &reg_changes, &new_was_changes](int src_id, int dst_id, int local_edge_pos,
                    long long int global_edge_pos, int vector_index)
            {
                float weight = outgoing_weights[global_edge_pos];
                float dst_weight = _distances[dst_id];
                float src_weight = _distances[src_id];
                if(dst_weight > src_weight + weight)
                {
                    _distances[dst_id] = src_weight + weight;
                    //new_was_changes[dst_id] = 1;
                    //new_was_changes[src_id] = 1;
                    reg_changes[vector_index] = 1;
                }
            };

            operations.advance(_graph, frontier, edge_op);

            int local_changes = 0;
            for(int i = 0; i < VECTOR_LENGTH; i++)
                local_changes += reg_changes[i];

            #pragma omp atomic
            changes += local_changes;
        }
        double t_end = omp_get_wtime();
        compute_time += t_end - t_st;

        /*#pragma omp parallel for
        for(int i = 0; i < vertices_count; i++)
        {
            was_changes[i] = new_was_changes[i];
        }*/

        //frontier.set_frontier_flags(frontier_condition);

        /*cout << "advance perf: " << edges_count / ((t_end - t_st) * 1e6) << " MTEPS" << endl;
        cout << "advance time: " << 1000.0 * (t_end - t_st) << " ms" << endl;

        t_st = omp_get_wtime();
        //frontier.generate_frontier(_graph, frontier_condition);
        frontier.set_all_active();
        t_end = omp_get_wtime();

        cout << "generate front BW: " << (2.0 * sizeof(int))*vertices_count / ((t_end - t_st) * 1e9) << " GB/s" << endl;
        cout << "generate perf: " << edges_count / ((t_end - t_st) * 1e6) << " MTEPS" << endl;
        cout << "generate time: " << 1000.0 * (t_end - t_st) << " ms" << endl;*/

        if(changes == 0)
            break;
    }
    t2 = omp_get_wtime();
    cout << "compute time: " << compute_time*1000 << " ms" << endl;
    cout << "inner perf: " << edges_count / (compute_time * 1e6) << " MTEPS" << endl;

    delete []was_changes;
    delete []new_was_changes;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
