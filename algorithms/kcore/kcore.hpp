//
//  kcore.hpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 01/10/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef kcore_hpp
#define kcore_hpp

#define KEEP_VERTEX 1
#define REMOVE_VERTEX 0

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void KCore<_TVertexValue, _TEdgeWeight>::allocate_result_memory(int _vertices_count, int **_kcore_data)
{
    *_kcore_data = new int[_vertices_count];
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void KCore<_TVertexValue, _TEdgeWeight>::free_result_memory(int *_kcore_data)
{
    delete[] _kcore_data;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void KCore<_TVertexValue, _TEdgeWeight>::kcore_subgraph(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                        int *_kcore_degrees,
                                                        int _k)
{
    int vertices_count    = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count   ();
    long long *outgoing_ptrs    = _graph.get_outgoing_ptrs   ();
    int       *outgoing_ids     = _graph.get_outgoing_ids    ();
    int       *incoming_degrees = _graph.get_incoming_degrees();
    
    int *remove_status = new int[vertices_count];
    
    double t1 = omp_get_wtime();
    // init data
    for(int i = 0; i < vertices_count; i++)
    {
        remove_status[i] = KEEP_VERTEX;
        _kcore_degrees[i] = incoming_degrees[i];
    }
    
    // remove loops
    for(int src_id = vertices_count - 1; src_id >= 0; src_id--)
    {
        long long edge_start = outgoing_ptrs[src_id];
        int connections_count = outgoing_ptrs[src_id + 1] - outgoing_ptrs[src_id];
        for(int edge_pos = 0; edge_pos < connections_count; edge_pos++)
        {
            int dst_id = outgoing_ids[edge_start + edge_pos];
            if(src_id == dst_id)
                _kcore_degrees[src_id]--;
        }
    }
    
    // start algorithm
    int last_removed = vertices_count - 1;
    for(int src_id = vertices_count - 1; src_id >= 0; src_id--)
    {
        if((remove_status[src_id] == KEEP_VERTEX) && (_kcore_degrees[src_id] < _k))
        {
            remove_status[src_id] = REMOVE_VERTEX;
                
            long long edge_start = outgoing_ptrs[src_id];
            int connections_count = outgoing_ptrs[src_id + 1] - outgoing_ptrs[src_id];
            for(int edge_pos = 0; edge_pos < connections_count; edge_pos++)
            {
                int dst_id = outgoing_ids[edge_start + edge_pos];
                //_kcore_degrees[dst_id]--;
                _kcore_degrees[src_id]--;
            }
                
            int src_id = last_removed;
        }
    }
    double t2 = omp_get_wtime();
    
    int kcore_size = 0;
    for(int i = 0; i < vertices_count; i++)
    {
        //cout << i << ") " << _kcore_degrees[i] << " " << remove_status[i] << endl;
        if(remove_status[i] == REMOVE_VERTEX)
            _kcore_degrees[i] = 0;
        
        if(_kcore_degrees[i] > 0)
            kcore_size++;
    }
    
    cout << "KCore size: " << kcore_size << endl;
    cout << "KCore Performance: " << ((double)edges_count) / ((t2 - t1) * 1e6) << " MTEPS" << endl;
    
    delete []remove_status;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void KCore<_TVertexValue, _TEdgeWeight>::kcore_subgraph(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                        int *_kcore_degrees,
                                                        int _k)
{
    LOAD_VECTORISED_CSR_GRAPH_DATA(_graph)
    
    int *remove_status = new int[vertices_count];
    
    double t1 = omp_get_wtime();
    
    // init data
    //#pragma omp parallel for schedule(static)
    for(int i = 0; i < vertices_count; i++)
    {
        remove_status[i] = KEEP_VERTEX;
        _kcore_degrees[i] = outgoing_sizes[i];
    }
    
    // process first part
    for(int src_id = 0; src_id < number_of_vertices_in_first_part; src_id++)
    {
        long long edge_start = first_part_ptrs[src_id];
        int connections_count = first_part_sizes[src_id];
        
        //#pragma omp parallel for schedule(static, 1)
        for(int edge_pos = 0; edge_pos < connections_count; edge_pos++)
        {
            int dst_id = outgoing_ids[edge_start + edge_pos];
            if(src_id == dst_id)
                _kcore_degrees[dst_id]--;
        }
    }
    
    // process last part
    //#pragma omp parallel for schedule(static, 1)
    for(int cur_vector_segment = 0; cur_vector_segment < vector_segments_count; cur_vector_segment++)
    {
        int segment_first_vertex = cur_vector_segment * VECTOR_LENGTH + number_of_vertices_in_first_part;
        long long segement_edges_start = vector_group_ptrs[cur_vector_segment];
        int segment_connections_count  = vector_group_sizes[cur_vector_segment];
        
        for(int edge_pos = 0; edge_pos < segment_connections_count; edge_pos++)
        {
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                int src_id = segment_first_vertex + i;
                int dst_id = outgoing_ids[segement_edges_start + edge_pos * VECTOR_LENGTH + i];
                if(src_id == dst_id)
                    _kcore_degrees[dst_id]--;
            }
        }
    }
    
    int update_required[VECTOR_LENGTH];
    
    // start algorithm
    int last_modifed_segment = (vector_segments_count - 1);
    
    int changes = 0;
    //while(changes)
    //{
        for(int cur_vector_segment = (vector_segments_count - 1); cur_vector_segment >= 0; cur_vector_segment--)
        {
            int segment_first_vertex = cur_vector_segment * VECTOR_LENGTH + number_of_vertices_in_first_part;
            long long segement_edges_start = vector_group_ptrs[cur_vector_segment];
            int segment_connections_count  = vector_group_sizes[cur_vector_segment];
            
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                int src_id = segment_first_vertex + i;
                if((remove_status[src_id] == KEEP_VERTEX) && (_kcore_degrees[src_id] < _k))
                {
                    update_required[i] = 1;
                    if(_kcore_degrees[src_id] == 0)
                        remove_status[src_id] = REMOVE_VERTEX;
                }
                else
                {
                    update_required[i] = 0;
                }
            }
            
            for(int edge_pos = 0; edge_pos < segment_connections_count; edge_pos++)
            {
                for(int i = 0; i < VECTOR_LENGTH; i++)
                {
                    int src_id = segment_first_vertex + i;
                    int dst_id = outgoing_ids[segement_edges_start + edge_pos * VECTOR_LENGTH + i];
                    
                    if(update_required[i] == 1)
                    {
                        remove_status[src_id] = REMOVE_VERTEX;
                        _kcore_degrees[src_id]--;
                        
                        cur_vector_segment = last_modifed_segment;
                    }
                }
            }
        }
    //}
    
    double t2 = omp_get_wtime();
    
    int kcore_size = 0;
    for(int i = 0; i < vertices_count; i++)
    {
        //cout << i << ") " << _kcore_degrees[i] << " " << remove_status[i] << endl;
        if(remove_status[i] == REMOVE_VERTEX)
            _kcore_degrees[i] = 0;
        
        if(_kcore_degrees[i] > 0)
            kcore_size++;
    }
    
    cout << "KCore size: " << kcore_size << endl;
    cout << "KCore vector performance: " << ((double)edges_count) / ((t2 - t1) * 1e6) << " MTEPS" << endl;
    
    delete []remove_status;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void KCore<_TVertexValue, _TEdgeWeight>::maximal_kcore(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                       int *_kcore_degrees)
{
    int peel = 1;
    
    int vertices_count    = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count   ();
    long long    *outgoing_ptrs = _graph.get_outgoing_ptrs   ();
    int          *outgoing_ids  = _graph.get_outgoing_ids    ();
    
    int *flag = new int[vertices_count];
    
    double t1 = omp_get_wtime();
    
    int num_active = vertices_count;
    #pragma omp parallel for schedule(static, 1)
    for(int src_id = 0; src_id < vertices_count; src_id++)
    {
        flag[src_id] = 0;
        _kcore_degrees[src_id] = outgoing_ptrs[src_id + 1] - outgoing_ptrs[src_id];
    }
    
    int last_active_vertex = (vertices_count - 1);
    int iterations_count = 0;
    while(num_active > 0)
    {
        iterations_count++;
        int processed_count = 0;
        
        //#pragma omp parallel
        //{
            int processed_local = 0;
            
           // #pragma omp for schedule(static, 1)
            for(int src_id = last_active_vertex; src_id >= 0; src_id--)
            {
                if((flag[src_id] == 0) && (_kcore_degrees[src_id] <= peel))
                {
                    flag[src_id] = peel;
                    processed_local++;
                }
            }
            
            //#pragma omp atomic
            processed_count += processed_local;
        //}
        
        num_active -= processed_count;
        
        while(flag[last_active_vertex] > 0)
        {
            last_active_vertex--;
        }
        
        if(processed_count > 0)
        {
            #pragma omp parallel for schedule(static, 1)
            for(int src_id = last_active_vertex; src_id >= 0; src_id--)
            {
                if(flag[src_id] == peel)
                {
                    long long edge_start = outgoing_ptrs[src_id];
                    int connections_count = outgoing_ptrs[src_id + 1] - outgoing_ptrs[src_id];
                    for(int edge_pos = 0; edge_pos < connections_count; edge_pos++)
                    {
                        int dst_id = outgoing_ids[edge_start + edge_pos];
                        _kcore_degrees[dst_id]--;
                        _kcore_degrees[src_id]--;
                    }
                }
            }
        }
        peel++;
    }
    
    for(int i = 0; i < vertices_count; i++)
        cout << i << ") " << _kcore_degrees[i] << ", flag = " << flag[i] << " | old val = " << (outgoing_ptrs[i + 1] - outgoing_ptrs[i]) << endl;
    
    double t2 = omp_get_wtime();
    cout << "iterations_count: " << iterations_count << endl;
    cout << "KCore peel Performance: " << ((double)edges_count) / ((t2 - t1) * 1e6) << " MTEPS" << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void KCore<_TVertexValue, _TEdgeWeight>::calculate_kcore_sizes(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                               int *_kcore_data,
                                                               int &_kcore_vertices_count,
                                                               long long &_kcore_edges_count)
{
    int vertices_count       = _graph.get_vertices_count();
    long long *outgoing_ptrs = _graph.get_outgoing_ptrs ();
    int       *outgoing_ids  = _graph.get_outgoing_ids  ();
    
    int vertices_in_kcore = 0;
    long long edges_in_kcore = 0;
    
    for(int src_id = 0; src_id < vertices_count; src_id++)
    {
        if(_kcore_data[src_id] > 0)
            vertices_in_kcore++;
        long long edge_start = outgoing_ptrs[src_id];
        int connections_count = outgoing_ptrs[src_id + 1] - outgoing_ptrs[src_id];
        for(int edge_pos = 0; edge_pos < connections_count; edge_pos++)
        {
            int dst_id = outgoing_ids[edge_start + edge_pos];
            if((_kcore_data[src_id] > 0) && (_kcore_data[dst_id] > 0))
            {
                edges_in_kcore++;
            }
        }
    }
    
    _kcore_vertices_count = vertices_in_kcore;
    _kcore_edges_count = edges_in_kcore;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* kcore_hpp */
