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

#ifdef __USE_NEC_SX_AURORA__
struct IndirectlyAccessedData
{
public:
    float *private_data;
    float *data;
    int size;

    IndirectlyAccessedData(int _size)
    {
        size = _size;
        data = new float[size];
        private_data = new float[MAX_SX_AURORA_THREADS * CACHED_VERTICES * CACHE_STEP];

        #pragma omp parallel
        {}
    }

    ~IndirectlyAccessedData()
    {
        delete []data;
        delete []private_data;
    }

    inline float get(int index, int tid)
    {
        float result = 0;
        if(index < CACHED_VERTICES)
            result = private_data[/*tid * CACHED_VERTICES * CACHE_STEP + */index * CACHE_STEP];
        else
            result = data[index];
        return result;
    }

    inline void set(int index, float val, int tid)
    {
        if(index < CACHED_VERTICES)
            this->private_data[/*tid * CACHED_VERTICES * CACHE_STEP + */index * CACHE_STEP] = val;
        else
            this->data[index] = val;
    }

    inline float &operator[] (int index)
    {
        if(index < CACHED_VERTICES)
            return private_data[index * CACHE_STEP];
        else
            return data[index];
    }

    void print()
    {
        for(int i = 0; i < size; i++)
        {
            cout << data[i] << " ";
        }
        cout << endl;
    }
};

template <typename _TVertexValue, typename _TEdgeWeight>
void ShortestPaths<_TVertexValue, _TEdgeWeight>::lib_dijkstra(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                              int _source_vertex,
                                                              _TEdgeWeight *_distances)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    cout << "alloc" << endl;

    #pragma omp parallel
    {};

    int large_threshold_size = VECTOR_LENGTH*MAX_SX_AURORA_THREADS*16;
    int medium_threshold_size = VECTOR_LENGTH;

    // split graphs into parts
    int large_threshold_vertex = 0;
    int medium_threshold_vertex = 0;
    for(int src_id = 0; src_id < vertices_count - 1; src_id++)
    {
        int cur_size = outgoing_ptrs[src_id + 1] - outgoing_ptrs[src_id];
        int next_size = 0;
        if(src_id < (vertices_count - 2))
        {
            next_size = outgoing_ptrs[src_id + 2] - outgoing_ptrs[src_id + 1];
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

    auto sssp_edge_op = [&outgoing_weights, &_distances, &_graph](int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos) {
        float weight = outgoing_weights[global_edge_pos];
        float dst_weight = _distances[dst_id];

        if (_distances[src_id] > dst_weight + weight)
        {
            _distances[src_id] = dst_weight + weight;
        }
    };

    #pragma omp parallel for
    for(int i = 0; i < vertices_count; i++)
        _distances[i] = FLT_MAX;
    _distances[_source_vertex] = 0;

    for(int iter = 0; iter < 20; iter++)
    {
        double t_start = omp_get_wtime();
        double wall_time = 0;
        double t1, t2, work;

        t1 = omp_get_wtime();
        #pragma omp parallel
        {
            for (int front_pos = 0; front_pos < large_threshold_vertex; front_pos++)
            {
                int src_id = front_pos;
                long long int start = outgoing_ptrs[src_id];
                long long int end = outgoing_ptrs[src_id + 1];

                #pragma _NEC ivdep
                #pragma _NEC vovertake
                #pragma _NEC novob
                #pragma _NEC vector
                #pragma omp for schedule(static)
                for(int i = start; i < end; i++)
                {
                    int global_idx = i;
                    int dst_id = outgoing_ids[global_idx];
                    _TEdgeWeight weight = outgoing_weights[global_idx];

                    _TEdgeWeight dst_weight = _distances[dst_id];
                    _TEdgeWeight src_weight = _distances[src_id];

                    if(dst_weight > src_weight + weight)
                        _distances[dst_id] = src_weight + weight;
                }
            }
        }
        #ifdef __PRINT_DETAILED_STATS__
        t2 = omp_get_wtime();
        wall_time += t2 - t1;
        work = outgoing_ptrs[large_threshold_vertex];
        cout << "time: " << (t2 - t1) * 1000 << " ms" << endl;
        cout << 100.0 * work / edges_count << " % of edges at large region" << endl;
        cout << "part 1(large) BW " << " : " << ((sizeof(int) * 5.0) * work) / ((t2 - t1) * 1e9) << " GB/s" << endl << endl;
        #endif

        t1 = omp_get_wtime();
        #pragma omp parallel
        {
            #pragma omp for schedule(static, 8)
            for (int front_pos = large_threshold_vertex; front_pos < medium_threshold_vertex; front_pos ++)
            {
                int src_id = front_pos;
                long long int start = outgoing_ptrs[src_id];
                long long int end = outgoing_ptrs[src_id + 1];
                int connections_count = end - start;

                int *ids_ptr = &outgoing_ids[start];
                _TEdgeWeight *weights_ptr = &outgoing_weights[start];

                for(int edge_vec_pos = 0; edge_vec_pos < connections_count - VECTOR_LENGTH; edge_vec_pos += VECTOR_LENGTH)
                {
                    #pragma _NEC ivdep
                    #pragma _NEC vovertake
                    #pragma _NEC novob
                    #pragma _NEC vector
                    for(int i = 0; i < VECTOR_LENGTH; i++)
                    {
                        int global_idx = edge_vec_pos + i;
                        int dst_id = ids_ptr[global_idx];
                        _TEdgeWeight weight = weights_ptr[global_idx];

                        _TEdgeWeight dst_weight = _distances[dst_id];
                        _TEdgeWeight src_weight = _distances[src_id];

                        if(dst_weight > src_weight + weight)
                            _distances[dst_id] = src_weight + weight;
                    }
                }

                #pragma _NEC ivdep
                #pragma _NEC vovertake
                #pragma _NEC novob
                #pragma _NEC vector
                for(int i = connections_count - VECTOR_LENGTH; i < connections_count; i++)
                {
                    int global_idx = i;
                    int dst_id = ids_ptr[global_idx];
                    _TEdgeWeight weight = weights_ptr[global_idx];

                    _TEdgeWeight dst_weight = _distances[dst_id];
                    _TEdgeWeight src_weight = _distances[src_id];
                    if(dst_weight > src_weight + weight)
                        _distances[dst_id] = src_weight + weight;
                }
            }
        }
        #ifdef __PRINT_DETAILED_STATS__
        t2 = omp_get_wtime();
        wall_time += t2 - t1;
        work = outgoing_ptrs[medium_threshold_vertex] - outgoing_ptrs[large_threshold_vertex];
        cout << "time: " << (t2 - t1) * 1000 << " ms" << endl;
        cout << 100.0 * work / edges_count << " % of edges at medium region" << endl;
        cout << "part 2(medium) BW " << " : " << ((sizeof(int) * 5.0) * work) / ((t2 - t1) * 1e9) << " GB/s" << endl << endl;
        #endif

        t1 = omp_get_wtime();
        #pragma omp parallel
        {
            long long int reg_start[VECTOR_LENGTH];
            long long int reg_end[VECTOR_LENGTH];
            int reg_connections[VECTOR_LENGTH];

            #pragma _NEC vreg(reg_start)
            #pragma _NEC vreg(reg_end)
            #pragma _NEC vreg(reg_connections)

            #pragma omp for schedule(static, 8)
            for(int front_pos = medium_threshold_vertex; front_pos < vertices_count; front_pos += VECTOR_LENGTH)
            {
                for(int i = 0; i < VECTOR_LENGTH; i++)
                {
                    int src_id = front_pos + i;
                    if(src_id < vertices_count)
                    {
                        reg_start[i] = outgoing_ptrs[src_id];
                        reg_end[i] = outgoing_ptrs[src_id + 1];
                        reg_connections[i] = reg_end[i] - reg_start[i];
                    }
                    else
                    {
                        reg_start[i] = 0;
                        reg_end[i] = 0;
                        reg_connections[i] = 0;
                    }
                }

                int max_connections = outgoing_ptrs[front_pos + 1] - outgoing_ptrs[front_pos];

                for(int edge_pos = 0; edge_pos < max_connections; edge_pos++)
                {
                    #pragma _NEC ivdep
                    #pragma _NEC vovertake
                    #pragma _NEC novob
                    #pragma _NEC vector
                    for(int i = 0; i < VECTOR_LENGTH; i++)
                    {
                        int src_id = front_pos + i;
                        if((src_id < vertices_count) && (edge_pos < reg_connections[i]))
                        {
                            int dst_id = outgoing_ids[reg_start[i] + edge_pos];
                            _TEdgeWeight weight = outgoing_weights[reg_start[i] + edge_pos];

                            _TEdgeWeight dst_weight = _distances[dst_id];
                            _TEdgeWeight src_weight = _distances[src_id];

                            if(dst_weight > src_weight + weight)
                                _distances[dst_id] = src_weight + weight;
                        }
                    }
                }
            }
        }
        #ifdef __PRINT_DETAILED_STATS__
        t2 = omp_get_wtime();
        wall_time += t2 - t1;
        work = edges_count - outgoing_ptrs[medium_threshold_vertex];
        cout << "time: " << (t2 - t1) * 1000 << " ms" << endl;
        cout << 100.0 * work / edges_count << " % of edges at small region" << endl;
        cout << "part 3(medium) BW " << " : " << ((sizeof(int) * 5.0) * work) / ((t2 - t1) * 1e9) << " GB/s" << endl << endl;
        #endif
        double t_end = omp_get_wtime();
        cout << "iter perf: " << (edges_count) / ((t_end - t_start) * 1e6) << " MTEPS" << endl;
        cout << "iter BW " << " : " << ((sizeof(int) * 5.0) * edges_count) / ((t_end - t_start) * 1e9) << " GB/s" << endl << endl;
    }

    cout << "large count: " << large_threshold_vertex << " | " << 100.0*(large_threshold_vertex)/vertices_count << endl;
    cout << "medium_count: " << medium_threshold_vertex - large_threshold_vertex << " | " << 100.0*(medium_threshold_vertex - large_threshold_vertex)/vertices_count << endl;
    cout << "small count: " << vertices_count - medium_threshold_vertex << " | " << 100.0*(vertices_count - medium_threshold_vertex)/vertices_count << endl;
    cout << "sssp one iter" << endl;
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
