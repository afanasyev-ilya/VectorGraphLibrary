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
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    GraphPrimitivesNEC operations;

    int large_threshold_size = VECTOR_LENGTH*MAX_SX_AURORA_THREADS*16;
    int medium_threshold_size = VECTOR_LENGTH;

    // split graphs into parts
    int large_threshold_vertex = 0;
    int medium_threshold_vertex = 0;

    double t1 = omp_get_wtime();
    #pragma omp parallel for
    for(int src_id = 0; src_id < vertices_count - 1; src_id++)
    {
        int cur_size = outgoing_ptrs[src_id + 1] -  outgoing_ptrs[src_id];
        int next_size = 0;
        if(src_id < (vertices_count - 2))
        {
            next_size = outgoing_ptrs[src_id + 2] -  outgoing_ptrs[src_id + 1];
        }
        if((cur_size >= large_threshold_size) && (next_size < large_threshold_size))
        {
            large_threshold_vertex = src_id;
        }

        if((cur_size >= medium_threshold_size) && (next_size < medium_threshold_size))
        {
            medium_threshold_vertex = src_id;
        }
    }
    double t2 = omp_get_wtime();
    cout << "split time: " << t2 - t1 << endl;

    FrontierNEC frontier(_graph.get_vertices_count());
    frontier.set_all_active();

    auto init_op = [_distances, _source_vertex] (int src_id)
    {
        if(src_id == _source_vertex)
            _distances[_source_vertex] = 0;
        else
            _distances[src_id] = FLT_MAX;
    };

    #pragma omp parallel
    {
        operations.init(_graph.get_vertices_count(), init_op);
    }

    int changes = 1;
    while(changes)
    {
        changes = 0;
        #pragma omp parallel
        {
            NEC_REGISTER_INT(changes, 0);
            NEC_REGISTER_FLT(distances, 0);

            auto edge_op = [&outgoing_weights, &_distances, &reg_changes](int src_id, int dst_id, int local_edge_pos,
                    long long int global_edge_pos, int vector_index)
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

            /*auto edge_op = [&outgoing_weights, &_distances, &reg_changes, &reg_distances](int src_id, int dst_id, int local_edge_pos,
                                                                                          long long int global_edge_pos, int vector_index)
            {
                float weight = outgoing_weights[global_edge_pos];
                float dst_weight = _distances[dst_id];
                float src_weight = _distances[src_id];
                if(src_weight > dst_weight + weight)
                {
                    reg_distances[vector_index] = dst_weight + weight;
                    //reg_changes[vector_index] = 1;
                }
            };*/

            operations.advance(_graph, frontier, large_threshold_vertex, medium_threshold_vertex, edge_op);

            int local_changes = 0;
            for(int i = 0; i < VECTOR_LENGTH; i++)
                local_changes += reg_changes[i];

            #pragma omp atomic
            changes += local_changes;
        }

        if(changes == 0)
            break;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
