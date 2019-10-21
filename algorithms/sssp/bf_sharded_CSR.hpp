//
//  bf_sharded_CSR.hpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 09/08/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef bf_sharded_CSR_hpp
#define bf_sharded_CSR_hpp

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
double ShortestPaths<_TVertexValue, _TEdgeWeight>::process_csr_shard(ShardedGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                                     _TEdgeWeight *_distances,
                                                                     _TEdgeWeight *_local_distances,
                                                                     int _shard_pos,
                                                                     int &_changes)
{
    ShardCSR<_TEdgeWeight> **shards_data = (ShardCSR<_TEdgeWeight> **)_graph.get_shards_data();
    int threads_count = omp_get_max_threads();
    
    // get shard data
    ShardCSRPointerData<_TEdgeWeight> shard_ptrs = shards_data[_shard_pos]->get_pointers_data();
    int vertices_in_shard = shards_data[_shard_pos]->get_vertices_in_shard();
    int *global_src_ids = shards_data[_shard_pos]->get_global_src_ids();
    
    // gather data
    shards_data[_shard_pos]->gather_local_shard_data(_local_distances, _distances);
    
    double t3 = omp_get_wtime();
    
    // process shard
    #pragma omp parallel num_threads(threads_count) shared(_changes)
    {
        int local_changes = 0;
        #pragma omp for schedule(static)
        for(int src_id = 0; src_id < vertices_in_shard; src_id++)
        {
            _TEdgeWeight current_distance = _local_distances[src_id];
            
            for(long long i = shard_ptrs.vertex_ptrs[src_id]; i < shard_ptrs.vertex_ptrs[src_id + 1]; i++)
            {
                int dst_id = shard_ptrs.dst_ids[i];
                _TEdgeWeight weight = shard_ptrs.weights[i];
                
                _TEdgeWeight dst_distance = _distances[dst_id] + weight;
                if(current_distance > dst_distance)
                {
                    current_distance = dst_distance;
                }
            }
            
            if(current_distance < _local_distances[src_id])
            {
                _local_distances[src_id] = current_distance;
                local_changes = 1;
            }
        }
        
        #pragma omp atomic
        _changes += local_changes;
    }
    double t4 = omp_get_wtime();
    
    // scatter data
    shards_data[_shard_pos]->scatter_local_shard_data(_local_distances, _distances);
    
    return t4 - t3;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
double ShortestPaths<_TVertexValue, _TEdgeWeight>::process_vect_csr_shard(ShardedGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                                          _TEdgeWeight *_distances,
                                                                          _TEdgeWeight *_local_distances,
                                                                          int _shard_pos,
                                                                          int &_changes)
{
    ShardVectCSR<_TEdgeWeight> **shards_data = (ShardVectCSR<_TEdgeWeight> **)_graph.get_shards_data();
    int threads_count = omp_get_max_threads();
    
    // get shard data
    ShardVectCSRPointerData<_TEdgeWeight> shard_ptrs = shards_data[_shard_pos]->get_pointers_data();
    int vertices_in_shard = shards_data[_shard_pos]->get_vertices_in_shard();
    int *global_src_ids = shards_data[_shard_pos]->get_global_src_ids();
    int *dst_ids = shard_ptrs.dst_ids;
    _TEdgeWeight *weights = shard_ptrs.weights;
    
    // gather data
    shards_data[_shard_pos]->gather_local_shard_data(_local_distances, _distances);
    
    double t3 = omp_get_wtime();
    
    // process shard
    #pragma omp parallel num_threads(threads_count) shared(_changes)
    {
        _TEdgeWeight reg_distances[VECTOR_LENGTH_IN_SHARD];
        int reg_changes[VECTOR_LENGTH_IN_SHARD];
        
        #pragma _NEC vreg(reg_distances)
        #pragma _NEC vreg(reg_changes)
        
        #pragma _NEC vector
        for(int i = 0; i < VECTOR_LENGTH_IN_SHARD; i++)
        {
            reg_changes[i] = 0;
        }
        
        int vector_segments_count = (shard_ptrs.vertices_in_shard - 1) / VECTOR_LENGTH_IN_SHARD + 1;
        
        #pragma omp for schedule(static, 1)
        for(int cur_vector_segment = 0; cur_vector_segment < vector_segments_count; cur_vector_segment++)
        {
            int segment_first_vertex = cur_vector_segment * VECTOR_LENGTH_IN_SHARD;
            long long edge_start = shard_ptrs.vector_group_ptrs[cur_vector_segment];
            int cur_max_connections_count = shard_ptrs.vector_group_sizes[cur_vector_segment];
            
            #pragma _NEC vector
            for(int i = 0; i < VECTOR_LENGTH_IN_SHARD; i++)
            {
                int src_id = segment_first_vertex + i;
                if(src_id < shard_ptrs.vertices_in_shard)
                {
                    reg_distances[i] = _local_distances[src_id];
                }
            }
            
            for(int edge_pos = 0; edge_pos < cur_max_connections_count; edge_pos++)
            {
                #pragma simd
                #pragma _NEC ivdep
                #pragma _NEC vovertake
                #pragma _NEC novob
                #pragma _NEC vector
                for(int i = 0; i < VECTOR_LENGTH_IN_SHARD; i++)
                {
                    int src_id = segment_first_vertex + i;
                    if(src_id < shard_ptrs.vertices_in_shard)
                    {
                        int dst_id = dst_ids[edge_start + edge_pos * VECTOR_LENGTH_IN_SHARD + i];
                        _TEdgeWeight weight = weights[edge_start + edge_pos * VECTOR_LENGTH_IN_SHARD + i];
                        _TEdgeWeight dst_distance = _distances[dst_id] + weight;
                        
                        if(reg_distances[i] > dst_distance)
                        {
                            reg_distances[i] = dst_distance;
                            reg_changes[i] = 1;
                        }
                    }
                }
            }
            
            #pragma _NEC vector
            for(int i = 0; i < VECTOR_LENGTH_IN_SHARD; i++)
            {
                int src_id = segment_first_vertex + i;
                if(src_id < shard_ptrs.vertices_in_shard)
                {
                    _local_distances[src_id] = reg_distances[i];
                }
            }
        }
        
        #pragma omp barrier
        
        int local_changes = 0;
        #pragma _NEC vector
        for(int i = 0; i < VECTOR_LENGTH_IN_SHARD; i++)
        {
            local_changes += reg_changes[i];
        }
        
        #pragma omp barrier
        
        #pragma omp atomic
        _changes += local_changes;
        
        #pragma omp barrier
    }
    double t4 = omp_get_wtime();
    
    // scatter data
    shards_data[_shard_pos]->scatter_local_shard_data(_local_distances, _distances);
    
    return t4 - t3;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ShortestPaths<_TVertexValue, _TEdgeWeight>::bellman_ford(ShardedGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                              int _source_vertex,
                                                              _TEdgeWeight *_distances)
{
    int number_of_shards = _graph.get_number_of_shards();
    
    int vertices_count = _graph.get_vertices_count();
    int threads_count = omp_get_max_threads();
    _TEdgeWeight *local_distances = _graph.template allocate_local_shard_data<_TEdgeWeight>();
    
    double t1 = omp_get_wtime();
    
    #pragma omp parallel for num_threads(threads_count)
    for(int i = 0; i < vertices_count; i++)
    {
        _distances[i] = FLT_MAX;
    }
    _distances[_source_vertex] = 0;
    
    double shard_compute_time = 0;
    
    int changes = 1;
    int iterations_count = 0;
    while(changes)
    {
        changes = 0;
        for(int shard_pos = 0; shard_pos < number_of_shards; shard_pos++)
        {
            if(_graph.get_shard_type() == SHARD_CSR_TYPE)
            {
                shard_compute_time += process_csr_shard(_graph, _distances, local_distances, shard_pos, changes);
            }
            else
            {
                shard_compute_time += process_vect_csr_shard(_graph, _distances, local_distances, shard_pos, changes);
            }
        }
        iterations_count++;
    }
    
    double t2 = omp_get_wtime();
    cout << "sharded BF total time: " << t2 - t1 << " s" << endl;
    cout << "shard compute time: " << shard_compute_time << " s" << endl;
    cout << "sharded BF Wall perf: " << ((double)_graph.get_edges_count()) / ((t2 - t1) * 1e6) << " MFLOPS" << endl;
    cout << "Perf per iteration: " << iterations_count * ((double)_graph.get_edges_count()) / (shard_compute_time * 1e6) << " MFLOPS" << endl;
    cout << "iterations count: " << iterations_count << endl << endl;
    
    delete []local_distances;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _TVertexValue, typename _TEdgeWeight>
void ShortestPaths<_TVertexValue, _TEdgeWeight>::gpu_bellman_ford(ShardedGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                                  int _source_vertex,
                                                                  _TEdgeWeight *_distances)
{
    _TEdgeWeight *device_distances;
    int vertices_count = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count();
    SAFE_CALL(cudaMalloc((void**)&device_distances, vertices_count * sizeof(_TEdgeWeight)));
    
    double t1, t2;
    int iterations_count = 0;
    int number_of_shards = _graph.get_number_of_shards();
    if(_graph.get_shard_type() == SHARD_CSR_TYPE)
    {
        ShardCSR<_TEdgeWeight> **shards_data = (ShardCSR<_TEdgeWeight> **)_graph.get_shards_data();
        ShardCSRPointerData<_TEdgeWeight> *device_shrads_data;
        cudaMallocManaged((void**)&device_shrads_data, number_of_shards*sizeof(ShardCSRPointerData<_TEdgeWeight>*));
        
        for(int shard_pos = 0; shard_pos < number_of_shards; shard_pos++)
        {
            device_shrads_data[shard_pos] = shards_data[shard_pos]->get_pointers_data();
        }
        
        t1 = omp_get_wtime();
        gpu_sharded_bellman_ford_wrapper(number_of_shards, (void*)device_shrads_data, device_distances, vertices_count,
                                         edges_count, iterations_count, _graph.get_shard_type() );
        
        t2 = omp_get_wtime();
        
        SAFE_CALL(cudaFree(device_shrads_data));
    }
    else if (_graph.get_shard_type() == SHARD_VECT_CSR_TYPE)
    {
        ShardVectCSR<_TEdgeWeight> **shards_data = (ShardVectCSR<_TEdgeWeight> **)_graph.get_shards_data();
        ShardVectCSRPointerData<_TEdgeWeight> *device_shrads_data;
        cudaMallocManaged((void**)&device_shrads_data, number_of_shards*sizeof(ShardVectCSRPointerData<_TEdgeWeight>*));
        
        for(int shard_pos = 0; shard_pos < number_of_shards; shard_pos++)
        {
            device_shrads_data[shard_pos] = shards_data[shard_pos]->get_pointers_data();
        }
        
        t1 = omp_get_wtime();
        gpu_sharded_bellman_ford_wrapper(number_of_shards, (void*)device_shrads_data, device_distances, vertices_count,
                                         edges_count, iterations_count, _graph.get_shard_type() );
        
        t2 = omp_get_wtime();
        
        SAFE_CALL(cudaFree(device_shrads_data));
    }
    
    cout << "sharded GPU time: " << t2 - t1 << endl;
    cout << "sharded GPU Perf: " << ((double)edges_count) / ((t2 - t1) * 1e6) << " MFLOPS" << endl;
    cout << "sharded GPU iterations count: " << iterations_count << endl;
    cout << "sharded GPU Perf per iteration: " << iterations_count * ((double)edges_count) / ((t2 - t1) * 1e6) << " MFLOPS" << endl;
    cout << "sharded GPU bandwidth: " << ((double)iterations_count)*((double)edges_count * (sizeof(int) + 2*sizeof(_TEdgeWeight))) / ((t2 - t1) * 1e9) << " gb/s" << endl << endl;
    
    SAFE_CALL(cudaMemcpy(_distances, device_distances, vertices_count * sizeof(_TEdgeWeight), cudaMemcpyDeviceToHost));

    SAFE_CALL(cudaFree(device_distances));
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* bf_sharded_CSR_hpp */
