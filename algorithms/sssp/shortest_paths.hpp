//
//  shortest_paths.hpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 18/04/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef shortest_paths_hpp
#define shortest_paths_hpp

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
class IndirectlyAccessedData
{
private:
    float *private_data;
    float *data;
    int size;
public:
    IndirectlyAccessedData(int _size)
    {
        size = _size;
        data = new float[size];
        private_data = new float[MAX_SX_AURORA_THREADS * CACHED_VERTICES * CACHE_STEP];

        #pragma omp parallel
        {};
    }

    ~IndirectlyAccessedData()
    {
        delete []data;
        delete []private_data;
    }

    inline float get(int index)
    {
        float result = 0;
        if(index < CACHED_VERTICES)
            result = private_data[index * CACHE_STEP];
        else
            result = data[index];
        return result;
    }

    inline void set(int index, float val)
    {
        if(index < CACHED_VERTICES)
            private_data[index * CACHE_STEP] = val;
        else
            data[index] = val;
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
void ShortestPaths<_TVertexValue, _TEdgeWeight>::lib_bellman_ford(
        ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
        int _source_vertex,
        _TEdgeWeight *_distances)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    _TEdgeWeight *cached_distances = _graph.template allocate_private_caches<_TEdgeWeight>(8);

    int large_threshold_size = 256*256;
    int medium_threshold_size = 256;

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
            medium_threshold_size = src_id;
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

    cout << "hey!" << endl;
    int iterations_count = 0;
    int changes = 1;
    while(changes > 0)
    {
        #pragma omp parallel num_threads(8)
        {
            _TEdgeWeight *private_distances = _graph.template get_private_data_pointer<_TEdgeWeight>(cached_distances);
            _graph.template place_data_into_cache<_TEdgeWeight>(_distances, private_distances);
        }

        changes = 0;

        double t1 = omp_get_wtime();
        for (int src_id = 0; src_id < large_threshold_vertex; src_id++)
        {
            long long int start = outgoing_ptrs[src_id];
            long long int end = outgoing_ptrs[src_id + 1];

            #pragma omp parallel
            {
                int reg_changes[VECTOR_LENGTH];
                #pragma _NEC vreg(reg_changes)
                for (int i = 0; i < VECTOR_LENGTH; i++)
                {
                    reg_changes[i] = 0;
                }

                _TEdgeWeight *private_distances = _graph.template get_private_data_pointer<_TEdgeWeight>(
                        cached_distances);

                #pragma omp for schedule(static)
                for (int i = start; i < end; i += VECTOR_LENGTH)
                {
                    #pragma _NEC ivdep
                    #pragma _NEC vovertake
                    #pragma _NEC novob
                    #pragma _NEC vector
                    #pragma _NEC cncall
                    for (int j = 0; j < VECTOR_LENGTH; j++)
                    {
                        int dst_id = outgoing_ids[i + j];

                        //sssp_edge_op(src_id, dst_id, i+j, i+j);
                        _TEdgeWeight weight = outgoing_weights[i+j];
                        _TEdgeWeight dst_weight = opt_distances.get(dst_id);//_graph.template load_vertex_data_cached<_TEdgeWeight>(dst_id, _distances, private_distances);

                        if (dst_weight > opt_distances.get(src_id) + weight)
                        {
                            opt_distances.set(dst_id, opt_distances.get(src_id) + weight);
                            reg_changes[j] = 1;
                        }
                    }
                }

                int local_changes = 0;
                for (int i = 0; i < VECTOR_LENGTH; i++)
                {
                    local_changes += reg_changes[i];
                }

                #pragma omp atomic
                changes += local_changes;
            }
        }
        double t2 = omp_get_wtime();
        double wall_time = t2 - t1;
        double work = outgoing_ptrs[large_threshold_vertex];
        cout << "part 1 changes: " << changes << endl;
        cout << "time: " << (t2 - t1) * 1000 << " ms" << endl;
        cout << "part 1(large) BW " << " : " << ((sizeof(int) * 4.0) * work) / ((t2 - t1) * 1e9) << " GB/s" << endl << endl;

        t1 = omp_get_wtime();
        #pragma omp parallel
        {
            int reg_changes[VECTOR_LENGTH];
            #pragma _NEC vreg(reg_changes)
            for (int i = 0; i < VECTOR_LENGTH; i++)
            {
                reg_changes[i] = 0;
            }

            _TEdgeWeight *private_distances = _graph.template get_private_data_pointer<_TEdgeWeight>(cached_distances);

            #pragma omp for schedule(static, 4)
            for (int front_pos = large_threshold_vertex; front_pos < medium_threshold_size; front_pos++)
            {
                int src_id = front_pos;//frontier_ids[front_pos];
                long long int start = outgoing_ptrs[src_id];
                long long int end = outgoing_ptrs[src_id + 1];

                #pragma _NEC ivdep
                #pragma _NEC vovertake
                #pragma _NEC novob
                #pragma _NEC vector
                #pragma _NEC cncall
                for (int i = start; i < end; i++)
                {
                    int dst_id = outgoing_ids[i];

                    _TEdgeWeight weight = outgoing_weights[i];
                    _TEdgeWeight dst_weight = opt_distances.get(dst_id);//_graph.template load_vertex_data_cached<_TEdgeWeight>(dst_id, _distances, private_distances);

                    if (dst_weight > opt_distances.get(src_id) + weight)
                    {
                        opt_distances.set(dst_id, opt_distances.get(src_id) + weight);
                        reg_changes[i % VECTOR_LENGTH] = 1;
                    }
                }
            }

            int local_changes = 0;
            for (int i = 0; i < VECTOR_LENGTH; i++)
            {
                local_changes += reg_changes[i];
            }

            #pragma omp atomic
            changes += local_changes;
        }
        t2 = omp_get_wtime();
        wall_time += t2 - t1;
        work = outgoing_ptrs[medium_threshold_size] - outgoing_ptrs[large_threshold_vertex];
        cout << "part 2 changes: " << changes << endl;
        cout << "time: " << (t2 - t1) * 1000 << " ms" << endl;
        cout << "part 2(medium) BW " << " : " << ((sizeof(int) * 3.0) * work) / ((t2 - t1) * 1e9) << " GB/s" << endl << endl;

        t1 = omp_get_wtime();
        #pragma omp parallel
        {
            _TEdgeWeight *private_distances = _graph.template get_private_data_pointer<_TEdgeWeight>(cached_distances);

            int reg_changes[VECTOR_LENGTH];
            #pragma _NEC vreg(reg_changes)
            for (int i = 0; i < VECTOR_LENGTH; i++)
            {
                reg_changes[i] = 0;
            }

            #pragma omp for schedule(static)
            for (int src_id = medium_threshold_size; src_id < vertices_count; src_id += VECTOR_LENGTH)
            {
                long long starts[VECTOR_LENGTH];
                int connections[VECTOR_LENGTH];

                int max_size = 0;
                for (int i = 0; i < VECTOR_LENGTH; i++)
                {
                    if((src_id + i) < vertices_count)
                    {
                        starts[i] = outgoing_ptrs[src_id + i];
                        connections[i] = outgoing_ptrs[src_id + i + 1] - outgoing_ptrs[src_id + i];
                    }
                    else
                    {
                        starts[i] = 0;
                        connections[i] = 0;
                    }
                }

                for (int i = 0; i < VECTOR_LENGTH; i++)
                {
                    if(max_size < connections[i])
                        max_size = connections[i];
                }

                for (int edge_pos = 0; edge_pos < max_size; edge_pos++)
                {
                    #pragma _NEC ivdep
                    #pragma _NEC vovertake
                    #pragma _NEC novob
                    #pragma _NEC vector
                    #pragma _NEC cncall
                    for (int i = 0; i < VECTOR_LENGTH; i++)
                    {
                        if ((edge_pos < connections[i]) && ((src_id + i) < vertices_count))
                        {
                            int dst_id = outgoing_ids[starts[i] + edge_pos];
                            _TEdgeWeight weight = outgoing_weights[starts[i] + edge_pos];
                            _TEdgeWeight dst_weight = opt_distances.get(dst_id);//_graph.template load_vertex_data_cached<_TEdgeWeight>(dst_id, _distances, private_distances);

                            if (dst_weight > opt_distances.get(src_id+i) + weight)
                            {
                                opt_distances.set(dst_id, opt_distances.get(src_id+i) + weight);
                                reg_changes[i] = 1;
                            }
                        }
                    }
                }
            }

            int local_changes = 0;
            for (int i = 0; i < VECTOR_LENGTH; i++)
            {
                local_changes += reg_changes[i];
            }

            #pragma omp atomic
            changes += local_changes;
        }
        t2 = omp_get_wtime();
        double first_part_time = wall_time;
        wall_time += t2 - t1;
        work = outgoing_ptrs[vertices_count] - outgoing_ptrs[medium_threshold_size];
        cout << "part 3 changes: " << changes << endl;
        cout << "time: " << (t2 - t1) * 1000 << " ms" << endl;
        cout << "part 3(small) BW " << " : " << ((sizeof(int) * 3.0) * work) / ((t2 - t1) * 1e9) << " GB/s" << endl
             << endl;
        /*cout << "wall BW: " << ((sizeof(int) * 3.0) * edges_count) / (wall_time * 1e9) << " GB/s" << endl;
        cout << "no reminder BW: "
             << ((sizeof(int) * 3.0) * outgoing_ptrs[medium_threshold_size]) / (first_part_time * 1e9) << " GB/s"
             << endl;*/
        cout << endl;

        iterations_count++;
    }

    cout << "ITERS DONE: " << iterations_count << endl;
    cout << "large count: " << large_threshold_vertex << " | " << 100.0*(large_threshold_vertex)/vertices_count << endl;
    cout << "medium_count: " << medium_threshold_size - large_threshold_vertex << " | " << 100.0*(medium_threshold_size - large_threshold_vertex)/vertices_count << endl;
    cout << "small count: " << vertices_count - medium_threshold_size << " | " << 100.0*(vertices_count - medium_threshold_size)/vertices_count << endl;
    cout << "VERSION : SMALL CHACHE V3 with step and core conflicts" << endl;

    for(int i = 0; i < vertices_count; i++)
    {
        _distances[i] = opt_distances[i];
    }
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* shortest_paths_hpp */
