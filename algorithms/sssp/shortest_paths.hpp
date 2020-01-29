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

    GraphPrimitivesNEC operations;

    int large_threshold_size = VECTOR_LENGTH*MAX_SX_AURORA_THREADS*16;
    int medium_threshold_size = VECTOR_LENGTH;

    // split graphs into parts
    int large_threshold_vertex = 0;
    int medium_threshold_vertex = 0;
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

    cout << "large_threshold_vertex: " << large_threshold_vertex << endl;

    #pragma omp parallel for
    for(int i = 0; i < vertices_count; i++)
        _distances[i] = FLT_MAX;
    _distances[_source_vertex] = 0;

    for(int iter = 0; iter < 20; iter++)
    {
        double t_start = omp_get_wtime();
        double wall_time = 0;
        double t1, t2, work;

        int changes = 0;

        #pragma omp parallel
        {
            int reg_changes[VECTOR_LENGTH];
            #pragma _NEC vreg(changes)
            for(int i = 0; i < VECTOR_LENGTH; i++)
                reg_changes[i] = 0;

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

            operations.advance(_graph, large_threshold_vertex, medium_threshold_vertex, edge_op);

            int local_changes = 0;
            for(int i = 0; i < VECTOR_LENGTH; i++)
                local_changes += reg_changes[i];

            #pragma omp atomic
            changes += local_changes;
        }
        double t_end = omp_get_wtime();
        cout << "changes: " << changes << endl;
        cout << "iter perf: " << (edges_count) / ((t_end - t_start) * 1e6) << " MTEPS" << endl;
        cout << "iter BW " << " : " << ((sizeof(int) * 5.0) * edges_count) / ((t_end - t_start) * 1e9) << " GB/s" << endl << endl;

        if(changes == 0)
            break;
    }

    cout << "large count: " << large_threshold_vertex << " | " << 100.0*(large_threshold_vertex)/vertices_count << endl;
    cout << "medium_count: " << medium_threshold_vertex - large_threshold_vertex << " | " << 100.0*(medium_threshold_vertex - large_threshold_vertex)/vertices_count << endl;
    cout << "small count: " << vertices_count - medium_threshold_vertex << " | " << 100.0*(vertices_count - medium_threshold_vertex)/vertices_count << endl;
    cout << "sssp one iter" << endl;
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
