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

    auto changes_on_prev_step = [&was_changes] (int src_id)->int {
        //return true;
        int res = NEC_NOT_IN_FRONTIER_FLAG;
        if(was_changes[src_id] > 0)
            res = NEC_IN_FRONTIER_FLAG;
        return res;
    };

    #pragma omp parallel
    {
        operations.init(vertices_count, init_op);
    }
    frontier.filter(_graph, changes_on_prev_step);

    t2 = omp_get_wtime();
    cout << "init time: " << (t2 - t1)*1000 << " ms" << endl;

    t1 = omp_get_wtime();
    int changes = 1;
    double compute_time = 0, filter_time = 0;
    int iterations_count = 0;
    for(int iter = 0; iter < vertices_count; iter++)
    {
        float *weights = outgoing_weights;
        if(frontier.type() == DENSE_FRONTIER)
            weights = ve_outgoing_weights;

        #pragma omp parallel for
        for(int i = 0; i < vertices_count; i++)
        {
            was_changes[i] = 0;
        }

        changes = 0;
        double t_st = omp_get_wtime();
        #pragma omp parallel
        {
            auto edge_op_push = [outgoing_weights, _distances, was_changes]
                    (int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos, int vector_index, int *safe_reg) {
                float weight = outgoing_weights[global_edge_pos];
                float dst_weight = _distances[dst_id];
                float src_weight = _distances[src_id];
                if(dst_weight > src_weight + weight)
                {
                    _distances[dst_id] = src_weight + weight;
                    was_changes[dst_id] = 1;
                    safe_reg[vector_index] = 1;
                }
            };

            auto edge_op_collective_push = [weights, _distances, was_changes]
                    (int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos, int vector_index, int *safe_reg){
                float weight = weights[global_edge_pos];
                float dst_weight = _distances[dst_id];
                float src_weight = _distances[src_id];
                if(dst_weight > src_weight + weight)
                {
                    _distances[dst_id] = src_weight + weight;
                    was_changes[dst_id] = 1;
                    was_changes[src_id] = 1;
                }
            };

            struct VertexPostprocessFunctor {
                int *was_changes;
                VertexPostprocessFunctor(int *_was_changes): was_changes(_was_changes) {}
                void operator()(int src_id, int connections_count, int *safe_reg)
                {
                    int res = 0;
                    #pragma _NEC vector
                    for(int i = 0; i < VECTOR_LENGTH; i++)
                        if(safe_reg[i] > 0)
                            res = 1;
                    if(res > 0)
                        was_changes[src_id] = res;
                }
            };
            VertexPostprocessFunctor vertex_postprocess_op(was_changes);

            operations.advance(_graph, frontier, edge_op_push, EMPTY_OP, vertex_postprocess_op, edge_op_collective_push,
                               USE_VECTOR_EXTENSION);
        }
        double t_end = omp_get_wtime();
        compute_time += t_end - t_st;

        t_st = omp_get_wtime();
        frontier.filter(_graph, changes_on_prev_step);
        t_end = omp_get_wtime();
        filter_time += t_end - t_st;

        if(frontier.size() == 0)
            break;
        iterations_count++;
    }
    t2 = omp_get_wtime();
    cout << "compute time: " << compute_time*1000 << " ms" << endl;
    cout << "filter time: " << filter_time*1000 << " ms" << endl;
    cout << "inner perf: " << edges_count / (compute_time * 1e6) << " MTEPS" << endl;
    cout << "wall perf: " << edges_count / ((compute_time + filter_time) * 1e6) << " MTEPS" << endl;
    cout << "iterations count: " << iterations_count << endl;
    cout << "perf per iteration: " << iterations_count * (edges_count / (compute_time * 1e6)) << " MTEPS" << endl;

    delete []was_changes;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
