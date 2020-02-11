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

struct VertexPostprocessFunctor {
    float * distances;
    const float * reg_distances;
    VertexPostprocessFunctor(float* _reg_distances, float *_distances): reg_distances(_reg_distances),distances(_distances) {}
    void operator()(int src_id, int connections_count) {
        float shortest_distance = FLT_MAX;

        #pragma _NEC shortloop_reduction
        #pragma _NEC nofuse
        #pragma _NEC vector
        #pragma _NEC ivdep
        for(int i = 0; i < 256; i++)
        {
            if(shortest_distance > reg_distances[i])
            {
                shortest_distance = reg_distances[i];
            }
        }

        if(src_id == 7533)
        {
            for(int i = 198; i < 256; i++)
                cout << "(" << reg_distances[i] << ")" << " ";
            cout << endl;
            cout << "min for 7533: " << shortest_distance << " , its connectiosn count: " << connections_count << endl;
        }

        if (shortest_distance < distances[src_id])
            distances[src_id] = shortest_distance;
    }
};

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

    auto all_active = [&was_changes] (int src_id)->int {
        return true;
        /*int res = 0;
        if(was_changes[src_id] > 0)
            res = 1;
        return res;*/
    };

    #pragma omp parallel
    {
        operations.init(vertices_count, init_op);
    }
    //frontier.generate_frontier(_graph, frontier_condition);
    frontier.filter(all_active);
    //frontier.set_frontier_flags(frontier_condition);

    t2 = omp_get_wtime();
    cout << "init time: " << (t2 - t1)*1000 << " ms" << endl;

    t1 = omp_get_wtime();
    int changes = 1;
    double compute_time = 0;
    int iterations_count = 0;
    for(int iter = 0; iter < vertices_count; iter++)
    {
        #pragma omp parallel for
        for(int i = 0; i < vertices_count; i++)
        {
            was_changes[i] = 0;
        }

        changes = 0;
        double t_st = omp_get_wtime();
        #pragma omp parallel num_threads(8)
        {
            NEC_REGISTER_INT(changes, 0);
            //NEC_REGISTER_FLT(distances, 0);
            float reg_distances[256];

            auto edge_op_push = [outgoing_weights, _distances, &reg_changes, &reg_distances]
                    (int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos, int vector_index) {
                float weight = outgoing_weights[global_edge_pos];
                float dst_weight = _distances[dst_id];
                //float src_weight = _distances[src_id];
                if(reg_distances[vector_index] > dst_weight + weight)
                {
                    //if(src_id == 7533)
                    //    cout << "update from edge " << dst_id << " , new val is " << dst_weight + weight << " vect pos: " << vector_index << " from edge pos " << local_edge_pos << endl;
                    //_distances[src_id] = dst_weight + weight;
                    reg_changes[vector_index] = 1;
                    reg_distances[vector_index] = dst_weight + weight;
                }
            };

            auto edge_op_collective_push = [ve_outgoing_weights, _distances, &reg_changes,&reg_distances]
                    (int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos, int vector_index){
                float weight = ve_outgoing_weights[global_edge_pos];
                float dst_weight = _distances[dst_id];
                float src_weight = _distances[src_id];
                if(src_weight > dst_weight + weight)
                {
                    _distances[src_id] = dst_weight + weight;
                    reg_changes[vector_index] = 1;
                }
            };

            struct VertexPreprocessFunctor {
                float * distances;
                float * const& reg_distances;
                VertexPreprocessFunctor(float* const& _reg_distances, float *_distances): reg_distances(_reg_distances),distances(_distances) {}
                inline void operator()(int src_id, int connections_count) const {
                    for(int i = 0; i < VECTOR_LENGTH; i++)
                    {
                        reg_distances[i] = distances[src_id];
                    }
                }
            };
            VertexPreprocessFunctor vertex_preprocess_op(reg_distances, _distances);


            VertexPostprocessFunctor vertex_postprocess_op(reg_distances, _distances);

            operations.advance(_graph, frontier, edge_op_push, vertex_preprocess_op, vertex_postprocess_op, edge_op_collective_push,
                               USE_VECTOR_EXTENSION);

            int local_changes = 0;
            for(int i = 0; i < VECTOR_LENGTH; i++)
                local_changes += reg_changes[i];
            #pragma omp atomic
            changes += local_changes;
        }
        double t_end = omp_get_wtime();
        compute_time += t_end - t_st;

        cout << "check after iter: " << _distances[7533] << endl;

        frontier.filter(all_active);

        if((frontier.size() == 0) || (changes == 0) || (iterations_count > 40))
            break;
        iterations_count++;
    }
    t2 = omp_get_wtime();
    cout << "compute time: " << compute_time*1000 << " ms" << endl;
    cout << "inner perf: " << edges_count / (compute_time * 1e6) << " MTEPS" << endl;
    cout << "iterations count: " << iterations_count << endl;
    cout << "perf per iteration: " << iterations_count * (edges_count / (compute_time * 1e6)) << " MTEPS" << endl;

    delete []was_changes;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
