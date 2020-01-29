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

    _TEdgeWeight *cached_distances = _graph.template allocate_private_caches<_TEdgeWeight>(8);
    cout << "alloc" << endl;
    _TEdgeWeight *result = new _TEdgeWeight[edges_count];
    _TEdgeWeight *tmp_distances = new _TEdgeWeight[vertices_count];

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

    IndirectlyAccessedData opt_distances(vertices_count);
    for(int i = 0; i < vertices_count; i++)
        opt_distances[i] = FLT_MAX;
    opt_distances[_source_vertex] = 0;

    for(int i = 0; i < vertices_count; i++)
        _distances[i] = FLT_MAX;
    _distances[_source_vertex] = 0;

    IndirectlyAccessedData *distances_ptr = &opt_distances;
    cout << "hey!" << endl;
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
                result[global_idx] = dst_id + weight;
            }
        }
    }
    t2 = omp_get_wtime();
    wall_time += t2 - t1;
    work = outgoing_ptrs[large_threshold_vertex];
    cout << "time: " << (t2 - t1) * 1000 << " ms" << endl;
    cout << 100.0 * work / edges_count << " % of edges at large region" << endl;
    cout << "part 1(large) BW " << " : " << ((sizeof(int) * 3.0) * work) / ((t2 - t1) * 1e9) << " GB/s" << endl << endl;

    t1 = omp_get_wtime();
    #pragma omp parallel
    {
        //int thread_id = omp_get_thread_num();
        int reg_result[VECTOR_LENGTH];

        #pragma _NEC vreg(reg_result)

        #pragma _NEC ivdep
        #pragma _NEC vector
        for(int i = 0; i < VECTOR_LENGTH; i++)
            reg_result[i] = 0;

        #pragma omp for schedule(static, 8)
        for (int front_pos = large_threshold_vertex; front_pos < medium_threshold_vertex; front_pos++)
        {
            int src_id = front_pos;
            long long int start = outgoing_ptrs[src_id];
            long long int end = outgoing_ptrs[src_id + 1];
            int connections_count = end - start;

            int *ids_ptr = &outgoing_ids[start];
            _TEdgeWeight *weights_ptr = &outgoing_weights[start];
            _TEdgeWeight *result_ptr = &result[start];

            //#pragma unroll()
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

                    if(src_weight > dst_weight + weight)
                        reg_result[i] = dst_weight + weight;
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
                if(src_weight > dst_weight + weight)
                    reg_result[i - (connections_count - 256)] = dst_weight + weight;
            }

            _TEdgeWeight shortest_distance = FLT_MAX;
            #pragma _NEC ivdep
            #pragma _NEC vector
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
               if(shortest_distance > reg_result[i])
                   shortest_distance = reg_result[i];
            }

            if(_distances[src_id] > shortest_distance)
            {
                _distances[src_id] = shortest_distance;
            }

            //result_ptr[i] = shortest_distance;

            // попробовать с разными массивами на чтения и запись, регистром
        }
    }
    t2 = omp_get_wtime();
    wall_time += t2 - t1;
    work = outgoing_ptrs[medium_threshold_vertex] - outgoing_ptrs[large_threshold_vertex];
    cout << "time: " << (t2 - t1) * 1000 << " ms" << endl;
    cout << 100.0 * work / edges_count << " % of edges at medium region" << endl;
    cout << "part 2(medium) BW " << " : " << ((sizeof(int) * 5.0) * work) / ((t2 - t1) * 1e9) << " GB/s" << endl << endl;

    cout << "large count: " << large_threshold_vertex << " | " << 100.0*(large_threshold_vertex)/vertices_count << endl;
    cout << "medium_count: " << medium_threshold_vertex - large_threshold_vertex << " | " << 100.0*(medium_threshold_vertex - large_threshold_vertex)/vertices_count << endl;
    cout << "small count: " << vertices_count - medium_threshold_vertex << " | " << 100.0*(vertices_count - medium_threshold_vertex)/vertices_count << endl;
    cout << "simple %, division, etc" << endl;

    for(int i = 0; i < 20; i++)
    {
        cout << result[i] << " ";
    }
    cout << endl;
    /*for(int i = 0; i < vertices_count; i++)
    {
        _distances[i] = opt_distances[i];
    }*/
    delete []result;
    delete []tmp_distances;
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
