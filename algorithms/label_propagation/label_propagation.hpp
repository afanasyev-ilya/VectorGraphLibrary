//
//  NEC_label_propagation.hpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 14/04/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef NEC_label_propagation_hpp
#define NEC_label_propagation_hpp

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void LabelPropagation<_TVertexValue, _TEdgeWeight>::allocate_result_memory(int _vertices_count, int **_result)
{
    *_result = new int[_vertices_count];
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void LabelPropagation<_TVertexValue, _TEdgeWeight>::free_result_memory(int *_result)
{
    delete[] _result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void LabelPropagation<_TVertexValue, _TEdgeWeight>::analyse_result(int *_labels, int _vertices_count)
{
    double *sorted_labels = new double[_vertices_count];
    
    #pragma omp parallel for
    for(int i = 0; i < _vertices_count; i++)
    {
        sorted_labels[i] = (double)_labels[i];
    }
    
    SortingAPI sorting_api;
    sorting_api.sort_array(sorted_labels, _vertices_count);
    cout << "sorted" << endl;
    
    map<int, int> communities_info;
    
    int last_pos = 0;
    int community_number = 1; // take into account last one
    for(int i = 1; i < _vertices_count; i++)
    {
        if(((int)sorted_labels[i]) != ((int)sorted_labels[i - 1]))
        {
            int current_community_size = i - last_pos;
            //cout << "Detected community with label " << (int)sorted_labels[i - 1] << " and size " << current_community_size << endl;
            communities_info[current_community_size]++;
            last_pos = i;
            community_number++;
        }
    }
    int current_community_size = _vertices_count - last_pos;
    //cout << "Detected community with label " << (int)sorted_labels[_vertices_count - 1] << " and size " << current_community_size << endl;
    communities_info[current_community_size]++;
    
    for(std::map<int, int>::iterator iter = communities_info.begin(); iter != communities_info.end(); ++iter)
    {
        cout << "there are " << iter->second << " communities of size: " << iter->first << endl;
    }
    cout << endl;
    
    cout << "Number of communities: " << community_number << endl;
    cout << "Average community size: " << _vertices_count/community_number << endl;
    
    delete []sorted_labels;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void LabelPropagation<_TVertexValue, _TEdgeWeight>::append_graph_with_labels(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int *_labels)
{
    for(int i = 0; i < _graph.get_vertices_count(); i++)
    {
        _graph.get_vertex_values()[i] = _labels[i];
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* NEC_label_propagation_hpp */
