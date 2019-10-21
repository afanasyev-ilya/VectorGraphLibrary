//
//  map_based_lp.hpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 21/04/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef map_based_lp_hpp
#define map_based_lp_hpp

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void LabelPropagation<_TVertexValue, _TEdgeWeight>::label_propagation_map_based(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int *_labels, int _threads_count)
{
    LOAD_VECTORISED_CSR_GRAPH_DATA(_graph);
    
    _graph.set_threads_count(_threads_count);
    
    int *old_labels = new int[vertices_count];
    
    #pragma omp parallel for num_threads(_threads_count)
    for(int i = 0; i < vertices_count; i++)
    {
        _labels[i] = i;
        old_labels[i] = i;
    }
    
    double t1 = omp_get_wtime();
    int changes = 1;
    int iterations_count = 0;
    #pragma omp parallel num_threads(_threads_count) shared(changes)
    {
        map<int, int> frequency_data;
        
        while(changes > 0)
        {
            #pragma omp barrier
            
            changes = 0;
            int private_changes = 0;
            
            // test simple
            #pragma omp for schedule(static, 1)
            for(int cur_vertex = 0; cur_vertex < vertices_count; cur_vertex++)
            {
                long long edge_pos = _graph.get_vector_connections_count(cur_vertex);
                int current_connections_count = _graph.get_vector_connections_count(cur_vertex);
                int step = VECTOR_LENGTH;
                if(cur_vertex < number_of_vertices_in_first_part)
                    step = 1;
                
                // traverse all adjacent edges
                for(int i = 0; i < current_connections_count; i++)
                {
                    int dst_id = outgoing_ids[edge_pos];
                    int dst_label = old_labels[dst_id];
                    
                    if(cur_vertex != dst_id) // if not a loop, TODO
                        frequency_data[dst_label]++;
                    
                    edge_pos += step;
                }
                
                int max_val = 0;
                int max_key = _labels[cur_vertex];
                for(map<int, int>::iterator iter = frequency_data.begin(); iter != frequency_data.end(); ++iter)
                {
                    if(iter->second > max_val)
                    {
                        max_key = iter->first;
                        max_val = iter->second;
                    }
                }
                
                if(max_key != _labels[cur_vertex])
                    private_changes++;
                
                _labels[cur_vertex] = max_key;
                
                frequency_data.clear();
            }
            
            #pragma omp barrier
            
            #pragma omp atomic
            changes += private_changes;
            
            #pragma omp master
            {
                iterations_count++;
            }
            
            #pragma omp for schedule(static, 1)
            for(int i = 0; i < vertices_count; i++)
            {
                old_labels[i] = _labels[i];
            }
            
            #pragma omp barrier
        }
    }
    double t2 = omp_get_wtime();
    cout << "time: " << t2 - t1 << endl;
    cout << "Perf: " << ((double)edges_count) / ((t2 - t1) * 1e6) << " MFLOPS" << endl;
    cout << "iterations count: " << iterations_count << endl;
    cout << "Perf per iteration: " << iterations_count * ((double)edges_count) / ((t2 - t1) * 1e6) << " MFLOPS" << endl;
    
    delete[] old_labels;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* map_based_lp_hpp */
