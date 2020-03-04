//
//  vect_dict_based_lp.hpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 21/04/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef vect_dict_based_lp_hpp
#define vect_dict_based_lp_hpp

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void LabelPropagation<_TVertexValue, _TEdgeWeight>::process_reminder_vertices(int *_outgoing_ids,
                                                                              int *_labels,
                                                                              long long int *_vector_group_ptrs,
                                                                              int *_vector_group_sizes,
                                                                              int _vector_segments_count,
                                                                              int _reminder_start,
                                                                              int _threads_count,
                                                                              int _max_dict_size,
                                                                              VectorDictionary **_vector_dicts)
{
    #pragma omp parallel num_threads(_threads_count)
    {
        int reg_labels[VECTOR_LENGTH];
        int reg_inc_required[VECTOR_LENGTH];
        
        #pragma vreg(reg_labels)
        #pragma vreg(reg_inc_required)
        
        int tid = omp_get_thread_num();
        VectorDictionary *frequency_data = _vector_dicts[tid];
        
        #pragma omp for schedule(static, 1)
        for(int cur_vector_segment = 0; cur_vector_segment < _vector_segments_count; cur_vector_segment++)
        {
            int segment_first_vertex = cur_vector_segment * VECTOR_LENGTH + _reminder_start;
            
            long long segement_edges_start = _vector_group_ptrs[cur_vector_segment];
            int segment_connections_count  = _vector_group_sizes[cur_vector_segment];
            
            #pragma unroll
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                int src_id = segment_first_vertex + i;
                reg_labels[i] = _labels[src_id];
            }
            
            frequency_data->clear();
            
            for(int edge_pos = 0; edge_pos < segment_connections_count; edge_pos++)
            {
                #pragma unroll
                for(int i = 0; i < VECTOR_LENGTH; i++)
                {
                    int src_id = segment_first_vertex + i;
                    int dst_id = _outgoing_ids[segement_edges_start + edge_pos * VECTOR_LENGTH + i];
                    int dst_label = _labels[dst_id];
                    
                    /*if(src_id != dst_id)
                        reg_inc_required[i] = 1;
                    else
                        reg_inc_required[i] = 0;*/
                }
                
                frequency_data->increment_values(reg_labels, reg_inc_required);
            }
            
            #pragma unroll
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                int src_id = segment_first_vertex + i;
                _labels[src_id] = reg_labels[i];
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void LabelPropagation<_TVertexValue, _TEdgeWeight>::label_propagation_vector_map_based(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int *_labels, int _threads_count)
{
    /*LOAD_VECTORISED_CSR_GRAPH_DATA(_graph);
    
    _graph.set_threads_count(_threads_count);
    
    // find max size
    int max_dict_size = 0;
    for(int cur_vector_segment = 0; cur_vector_segment < vector_segments_count; cur_vector_segment++)
    {
        int segment_connections_count  = vector_group_sizes[cur_vector_segment];
        if(segment_connections_count > max_dict_size)
            max_dict_size = segment_connections_count;
    }
    
    VectorDictionary **vector_dicts = new VectorDictionary*[_threads_count];
    for(int i = 0; i < _threads_count; i++)
        vector_dicts[i] = new VectorDictionary(max_dict_size);
        
    cout << "max dict size: " << max_dict_size << endl;
    
    #pragma retain(_labels)
    
    #pragma omp parallel num_threads(_threads_count)
    for(int i = 0; i < vertices_count; i++)
    {
        _labels[i] = i;
    }
    
    double t1 = omp_get_wtime();
    int iterations_count = 0;
    for(iterations_count = 0; iterations_count < 10; iterations_count++)
    {
        //process_first_vertices(...);
        process_reminder_vertices(outgoing_ids, _labels, vector_group_ptrs, vector_group_sizes, vector_segments_count,
                                  number_of_vertices_in_first_part, _threads_count, max_dict_size, vector_dicts);
    }
    double t2 = omp_get_wtime();
    
    cout << "time: " << t2 - t1 << endl;
    cout << "Perf: " << ((double)edges_count) / ((t2 - t1) * 1e6) << " MFLOPS" << endl;
    cout << "iterations count: " << iterations_count << endl;
    cout << "Perf per iteration: " << iterations_count * ((double)edges_count) / ((t2 - t1) * 1e6) << " MFLOPS" << endl;
    
    for(int i = 0; i < _threads_count; i++)
        delete vector_dicts[i];
    delete []vector_dicts;*/
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* vect_dict_based_lp_hpp */
