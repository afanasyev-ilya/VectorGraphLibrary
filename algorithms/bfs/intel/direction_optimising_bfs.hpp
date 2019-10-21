//
//  direction_optimising_bfs.hpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 30/06/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef intel_direction_optimising_bfs_hpp
#define intel_direction_optimising_bfs_hpp

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void BFS<_TVertexValue, _TEdgeWeight>::intel_top_down_step(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                           int *_outgoing_ids,
                                                           int _number_of_vertices_in_first_part,
                                                           int *_levels,
                                                           VertexQueue &_global_queue,
                                                           VertexQueue **_local_queues,
                                                           int _omp_threads,
                                                           int _current_level,
                                                           int &_vis,
                                                           int &_in_lvl)
{
    int heavy_vertices = 0;
    int current_frontier_size = _global_queue.get_size();
    int *current_frontier_data = _global_queue.get_data();
    vector<int> queue_for_large_vertices;
    for(int i = 0; i < current_frontier_size; i++)
    {
        int vertex_id = current_frontier_data[i];
        if(vertex_id < _number_of_vertices_in_first_part)
        {
            heavy_vertices++;
            queue_for_large_vertices.push_back(vertex_id);
        }
    }
    
    #pragma omp parallel shared(queue_for_large_vertices, _vis, _in_lvl) num_threads(_omp_threads)
    {
        int tid = omp_get_thread_num();
        VertexQueue *local_queue = _local_queues[tid];
        
        int local_vis = 0;
        int local_in_lvl = 0;
        
        for(int vertex_pos = 0; vertex_pos < heavy_vertices; vertex_pos++)
        {
            int src_id = queue_for_large_vertices[vertex_pos];
            
            long long edge_pos = _graph.get_vertex_pointer(src_id);
            int connections_count = _graph.get_vector_connections_count(src_id);
            
            #pragma omp for schedule(static)
            for(int i = 0; i < connections_count; i++)
            {
                local_in_lvl++;
                int dst_id = _outgoing_ids[edge_pos + i];
                
                if(_levels[dst_id] == -1)
                {
                    _levels[dst_id] = _current_level;
                    local_queue->push_back(dst_id);
                    local_vis++;
                }
            }
        }
        
        #pragma omp barrier
        
        #pragma omp atomic
        _vis += local_vis;
        
        #pragma omp atomic
        _in_lvl += local_in_lvl;
    }
    
    #pragma omp parallel shared(_global_queue, _vis, _in_lvl) num_threads(_omp_threads)
    {
        int tid = omp_get_thread_num();
        VertexQueue *local_queue = _local_queues[tid];
        
        int local_vis = 0;
        int local_in_lvl = 0;
        
        #pragma omp for schedule(static)
        for(int vertex_pos = 0; vertex_pos < current_frontier_size; vertex_pos++)
        {
            int src_id = current_frontier_data[vertex_pos];
            
            if(src_id < _number_of_vertices_in_first_part)
                continue;
            
            long long edge_pos = _graph.get_vertex_pointer(src_id);
            int connections_count = _graph.get_vector_connections_count(src_id);
            
            for(int i = 0; i < connections_count; i++)
            {
                local_in_lvl++;
                int dst_id = _outgoing_ids[edge_pos + i * VECTOR_LENGTH];
                
                if(_levels[dst_id] == -1)
                {
                    _levels[dst_id] = _current_level;
                    local_queue->push_back(dst_id);
                    local_vis++;
                }
            }
        }
        
        #pragma omp atomic
        _vis += local_vis;
        
        #pragma omp atomic
        _in_lvl += local_in_lvl;
    }
    
    #pragma omp parallel shared(_global_queue) num_threads(_omp_threads)
    {
        int tid = omp_get_thread_num();
        VertexQueue *local_queue = _local_queues[tid];
        _global_queue.append_with_local_queues(*local_queue);
        
        #pragma omp barrier
        
        local_queue->clear();
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void BFS<_TVertexValue, _TEdgeWeight>::intel_bottom_up_step(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                            long long *_first_part_ptrs,
                                                            int *_first_part_sizes,
                                                            int _vector_segments_count,
                                                            long long *_vector_group_ptrs,
                                                            int *_vector_group_sizes,
                                                            int *_outgoing_ids,
                                                            int _vertices_count,
                                                            int _number_of_vertices_in_first_part,
                                                            int *_levels,
                                                            VertexQueue &_global_queue,
                                                            VertexQueue **_local_queues,
                                                            int _omp_threads,
                                                            int _current_level,
                                                            int &_vis,
                                                            int &_in_lvl)
{
    #pragma omp parallel shared(_global_queue, _vis, _in_lvl) num_threads(_omp_threads)
    {
        int tid = omp_get_thread_num();
        VertexQueue *local_queue = _local_queues[tid];
        
        int local_vis = 0;
        int local_in_lvl = 0;
        
        #pragma omp for schedule(static, 1)
        for(int src_id = 0; src_id < _number_of_vertices_in_first_part; src_id++)
        {
            long long edge_start = _first_part_ptrs[src_id];
            int connections_count = _first_part_sizes[src_id];
            
            if(_levels[src_id] == -1)
            {
                for(int edge_pos = 0; edge_pos < connections_count; edge_pos++)
                {
                    local_in_lvl++;
                    int dst_id = _outgoing_ids[edge_start + edge_pos];
                    if(_levels[dst_id] == (_current_level - 1))
                    {
                        _levels[src_id] = _current_level;
                        local_queue->push_back(src_id);
                        local_vis++;
                        break;
                    }
                }
            }
        }
        
        #pragma omp for schedule(static, 1)
        for(int cur_vector_segment = 0; cur_vector_segment < _vector_segments_count; cur_vector_segment++)
        {
            int segment_first_vertex = cur_vector_segment * VECTOR_LENGTH + _number_of_vertices_in_first_part;
            
            long long segement_edges_start = _vector_group_ptrs[cur_vector_segment];
            int segment_connections_count  = _vector_group_sizes[cur_vector_segment];
            
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                int src_id = segment_first_vertex + i;
                if(_levels[src_id] == -1)
                {
                    for(int edge_pos = 0; edge_pos < segment_connections_count; edge_pos++)
                    {
                        local_in_lvl++;
                        int dst_id = _outgoing_ids[segement_edges_start + edge_pos * VECTOR_LENGTH + i];
                        if(_levels[dst_id] == (_current_level - 1))
                        {
                            _levels[src_id] = _current_level;
                            local_queue->push_back(src_id);
                            local_vis++;
                            break;
                        }
                    }
                }
            }
        }
        
        #pragma omp barrier
        
        #pragma omp atomic
        _vis += local_vis;
        
        #pragma omp atomic
        _in_lvl += local_in_lvl;
    }
    
    #pragma omp parallel shared(_global_queue) num_threads(_omp_threads)
    {
        int tid = omp_get_thread_num();
        VertexQueue *local_queue = _local_queues[tid];
        _global_queue.append_with_local_queues(*local_queue);
        
        #pragma omp barrier
        
        local_queue->clear();
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void BFS<_TVertexValue, _TEdgeWeight>::intel_direction_optimising_BFS(
                                                        VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                        int *_levels,
                                                        int _source_vertex)
{
    LOAD_VECTORISED_CSR_GRAPH_DATA(_graph)
    
    int omp_threads = omp_get_max_threads();
    _graph.set_threads_count(omp_threads);
    
    VertexQueue **local_queues = new VertexQueue *[omp_threads];
    for(int i = 0; i < omp_threads; i++)
        local_queues[i] = new VertexQueue(vertices_count);
    
    _graph.template vertex_array_set_to_constant<int>(_levels, -1);
    _graph.template vertex_array_set_element<int>(_levels, _source_vertex, 1);
    
    VertexQueue global_queue(vertices_count);
    global_queue.push_back(_source_vertex);
    int current_level = 2;
    
    vector<int> level_num;
    vector<double> level_perf;
    vector<string> level_state;
    vector<long long> level_edges_checked;
    
    double t1 = omp_get_wtime();
    StateOfBFS current_state = TOP_DOWN;
    while(!global_queue.empty())
    {
        double t3 = omp_get_wtime();
        
        int current_queue_size = global_queue.get_size();
        int vis = 0, in_lvl = 0;
        
        if(current_state == BOTTOM_UP)
        {
            level_state.push_back("BU");
            intel_bottom_up_step(_graph, first_part_ptrs, first_part_sizes, vector_segments_count, vector_group_ptrs,
                                 vector_group_sizes, outgoing_ids, vertices_count, number_of_vertices_in_first_part,
                                 _levels, global_queue, local_queues, omp_threads, current_level, vis, in_lvl);
        }
        else if(current_state == TOP_DOWN)
        {
            level_state.push_back("TD");
            intel_top_down_step(_graph, outgoing_ids, number_of_vertices_in_first_part, _levels, global_queue, local_queues,
                                omp_threads, current_level, vis, in_lvl);
        }
        
        current_level++;
        
        int next_queue_size = global_queue.get_size();
        
        current_state = change_state(current_queue_size, next_queue_size, vertices_count, edges_count,
                                     current_state, vis, in_lvl);
        
        double t4 = omp_get_wtime();
        level_num.push_back(current_level - 1);
        level_perf.push_back(t4 - t3);
        level_edges_checked.push_back(in_lvl);
        
    }
    double t2 = omp_get_wtime();
    
    cout << "BFS perf: " << ((double)edges_count) / ((t2 - t1) * 1e6) << " MFLOPS" << endl;
    double edges_sum = 0;
    for(int i = 0; i < level_perf.size(); i++)
    {
        cout << "level " << level_num[i] << " in " << level_state[i] <<  " | perf: " << ((double)edges_count) / (level_perf[i] * 1e6) << " MFLOPS | " << level_edges_checked[i] << endl;
        edges_sum += level_edges_checked[i];
    }
    cout << "total edges: " << edges_count << endl;
    cout << "BFS checked: " << (edges_sum / ((double)edges_count)) * 100.0 << " % of graph edges" << endl;
    
    for(int i = 0; i < omp_threads; i++)
        delete local_queues[i];
    delete []local_queues;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* intel_direction_optimising_bfs_hpp */
