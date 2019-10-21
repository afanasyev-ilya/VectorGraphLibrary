//
//  init_graph_helpers.hpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 06/05/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef init_graph_helpers_hpp
#define init_graph_helpers_hpp

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void VectorisedCSRGraph<_TVertexValue, _TEdgeWeight>::construct_tmp_graph_from_edges_list(
                                                    EdgesListGraph<_TVertexValue, _TEdgeWeight> &_old_graph,
                                                    vector<vector<TempEdgeData<_TEdgeWeight> > > &_tmp_graph,
                                                    vector<_TVertexValue> &_tmp_vertex_values,
                                                    int _tmp_vertices_count,
                                                    SupportedTraversalType _traversal_type)
{
    int *old_src_ids = _old_graph.get_src_ids();
    int *old_dst_ids = _old_graph.get_dst_ids();
    _TEdgeWeight *old_weights = _old_graph.get_weights();
    _TVertexValue *old_values = _old_graph.get_vertex_values();
    
    int tmp_vertices_count = _old_graph.get_vertices_count();
    long long int tmp_edges_count = _old_graph.get_edges_count();
    
    for(int i = 0; i < _tmp_vertices_count; i++)
    {
        vector<TempEdgeData<_TEdgeWeight> > empty_vector;
        _tmp_graph.push_back(empty_vector);
        _tmp_vertex_values.push_back(old_values[i]);
    }
    
    for(long long int i = 0; i < tmp_edges_count; i++)
    {
        int src_id = old_src_ids[i];
        int dst_id = old_dst_ids[i];
        _TEdgeWeight weight = old_weights[i];
        
        if(_traversal_type == PUSH_TRAVERSAL)
        {
            if((src_id < 0) || (src_id >= _tmp_vertices_count))
                throw "ERROR: bad src_id in construct_tmp_graph_from_edges_list";
            _tmp_graph[src_id].push_back(TempEdgeData<_TEdgeWeight>(dst_id, weight));
        }
        else if(_traversal_type == PULL_TRAVERSAL)
        {
            if((dst_id < 0) || (dst_id >= _tmp_vertices_count))
                throw "ERROR: bad dst_id in construct_tmp_graph_from_edges_list";
            _tmp_graph[dst_id].push_back(TempEdgeData<_TEdgeWeight>(src_id, weight));
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void VectorisedCSRGraph<_TVertexValue, _TEdgeWeight>::sort_vertices_in_descending_order(
                                                               vector<vector<TempEdgeData<_TEdgeWeight> > > &_tmp_graph,
                                                               int *_tmp_reordered_vertex_ids,
                                                               int _tmp_vertices_count)
{
    // sort all vertices now
    vector<pair<int, int> > pairs(_tmp_vertices_count);
    for(int i = 0; i < _tmp_vertices_count; ++i)
        pairs[i] = make_pair(_tmp_graph[i].size(), i);
    
    if(vertices_state == VERTICES_SORTED)
    {
        sort(pairs.begin(), pairs.end());
        reverse(pairs.begin(), pairs.end());
    }
    
    // save old indexes array
    int *old_indexes = new int[_tmp_vertices_count];
    for(int i = 0; i < _tmp_vertices_count; i++)
    {
        old_indexes[i] = pairs[i].second;
    }
    
    // need to reoerder all data arrays in 2 steps
    vector<vector<TempEdgeData<_TEdgeWeight> > > new_tmp_graph(_tmp_vertices_count);
    for(int i = 0; i < _tmp_vertices_count; i++)
    {
        new_tmp_graph[i] = _tmp_graph[old_indexes[i]];
    }
    
    for(int i = 0; i < _tmp_vertices_count; i++)
    {
        _tmp_graph[i] = new_tmp_graph[i];
    }
    
    // get correct reordered array
    for(int i = 0; i < _tmp_vertices_count; i++)
    {
        _tmp_reordered_vertex_ids[old_indexes[i]] = i;
    }
    delete []old_indexes;
    
    // set new values of reordered vertices into graph edges
    for(int cur_vertex = 0; cur_vertex < _tmp_vertices_count; cur_vertex++)
    {
        int src_id = cur_vertex;
        for(int i = 0; i < _tmp_graph[src_id].size(); i++)
        {
            _tmp_graph[src_id][i].dst_id = _tmp_reordered_vertex_ids[_tmp_graph[src_id][i].dst_id];
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void VectorisedCSRGraph<_TVertexValue, _TEdgeWeight>::sort_edges(vector<vector<TempEdgeData<_TEdgeWeight> > > &_tmp_graph,
                                                                 int _tmp_vertices_count)
{
    // sort adjacent ids locally for each vertex
    for(int cur_vertex = 0; cur_vertex < _tmp_vertices_count; cur_vertex++)
    {
        int src_id = cur_vertex;
        if(edges_state == EDGES_SORTED)
        {
            std::sort(_tmp_graph[src_id].begin(), _tmp_graph[src_id].end(), edge_cmp<_TEdgeWeight>);
        }
        else if(edges_state == EDGES_RANDOM_SHUFFLED)
        {
            std::random_shuffle(_tmp_graph[src_id].begin(), _tmp_graph[src_id].end());
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
int VectorisedCSRGraph<_TVertexValue, _TEdgeWeight>::calculate_and_find_threshold_vertex(
                                                    vector<vector<TempEdgeData<_TEdgeWeight> > > &_tmp_graph,
                                                    int _tmp_vertices_count)
{
    int threshold_vertex = 0;
    int typical_threads_count = 8;
    int threshold_value = VECTOR_LENGTH * typical_threads_count * 16;
    
    int large_vertices = 0;
    int small_vertices = 0;
    for(int i = 0; i < (_tmp_vertices_count - 1); i++)
    {
        if((_tmp_graph[i].size() > threshold_value) && (_tmp_graph[i + 1].size() <= threshold_value))
        {
            threshold_vertex = i + 1;
            break;
        }
    }
    cout << " we have " << threshold_vertex << " large vertices" << endl;
    cout << " we have " << _tmp_vertices_count - threshold_vertex << " small vertices" << endl;
    
    cout << "largest vertex degree: " << _tmp_graph[0].size() << endl;
    cout << "smallest vertex degree: " << _tmp_graph[threshold_vertex].size() << endl << endl;
    return (int)min(_tmp_vertices_count * 0.1, (double)threshold_vertex);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void VectorisedCSRGraph<_TVertexValue, _TEdgeWeight>::flatten_graph(
                                                    vector<vector<TempEdgeData<_TEdgeWeight> > > &_tmp_graph,
                                                    int &_tmp_vertices_count,
                                                    long long int &_tmp_edges_count,
                                                    long long int _old_edges_count,
                                                    int _last_part_start)
{
    // save old vertices count
    old_vertices_count = _tmp_vertices_count;
    
    // flatten first part of vertices
    number_of_vertices_in_first_part = _last_part_start;
    for(int i = 0; i < _last_part_start; i++)
    {
        while(_tmp_graph[i].size() % supported_vector_length != 0)
        {
            _tmp_graph[i].push_back(TempEdgeData<_TEdgeWeight>(i, 0.0));
            _tmp_edges_count++;
        }
    }
    
    // add vertices to have size of remonder % VECTOR_LENGTH = 0
    int additional_vertices_count = 0;
    while((_tmp_graph.size() - number_of_vertices_in_first_part) % supported_vector_length != 0)
    {
        vector<TempEdgeData<_TEdgeWeight> > empty_vector;
        _tmp_graph.push_back(empty_vector);
        additional_vertices_count++;
    }
    cout << "added " << additional_vertices_count << " vertices" << endl;
    
    // flatten reminder
    _tmp_vertices_count = _tmp_graph.size();
    for(int i = _last_part_start; i < _tmp_vertices_count; i += supported_vector_length)
    {
        int max_segment_size = _tmp_graph[i].size();
        for(int j = 0; j < supported_vector_length; j++)
        {
            while(_tmp_graph[i + j].size() < max_segment_size)
            {
                _tmp_graph[i + j].push_back(TempEdgeData<_TEdgeWeight>(i + j, 0.0));
                _tmp_edges_count++;
            }
        }
    }
    
    cout << "New graph is " << ((double)_tmp_edges_count) / ((double)_old_edges_count) << " times larger " << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void VectorisedCSRGraph<_TVertexValue, _TEdgeWeight>::convert_tmp_graph_into_vect_CSR(
                                                   vector<vector<TempEdgeData<_TEdgeWeight> > > &_tmp_graph,
                                                   int *_tmp_reordered_vertex_ids,
                                                   vector<_TVertexValue> &_tmp_vertex_values,
                                                   int _tmp_vertices_count)
{
    // convert fisrt part
    long long current_edge = 0;
    for(int cur_vertex = 0; cur_vertex < number_of_vertices_in_first_part; cur_vertex++)
    {
        int cur_connections_count = _tmp_graph[cur_vertex].size();
        
        first_part_ptrs[cur_vertex] = current_edge;
        first_part_sizes[cur_vertex] = cur_connections_count;
        
        for(int edge_pos = 0; edge_pos < cur_connections_count; edge_pos++)
        {
            outgoing_ids[current_edge] = _tmp_graph[cur_vertex][edge_pos].dst_id;
            outgoing_weights[current_edge] = _tmp_graph[cur_vertex][edge_pos].weight;
            current_edge++;
        }
    }
    
    for(int i = 0; i < old_vertices_count; i++)
    {
        this->vertex_values[i] = _tmp_vertex_values[i];
        this->reordered_vertex_ids[i] = _tmp_reordered_vertex_ids[i];
    }
    
    for(int i = old_vertices_count; i < this->vertices_count; i++)
    {
        this->vertex_values[i] = -1;
        this->reordered_vertex_ids[i] = i;
    }
    
    // convert last part
    for(int cur_vector_segment = 0; cur_vector_segment < vector_segments_count; cur_vector_segment++)
    {
        int vec_start = cur_vector_segment * supported_vector_length + number_of_vertices_in_first_part;
        
        int cur_max_connections_count = _tmp_graph[vec_start].size();
        
        vector_group_ptrs[cur_vector_segment] = current_edge;
        vector_group_sizes[cur_vector_segment] = cur_max_connections_count;
        
        for(int edge_pos = 0; edge_pos < cur_max_connections_count; edge_pos++)
        {
            #pragma unroll
            for(int i = 0; i < supported_vector_length; i++)
            {
                int src_id = vec_start + i;
                outgoing_ids[current_edge + i] = _tmp_graph[src_id][edge_pos].dst_id;
                outgoing_weights[current_edge + i] = _tmp_graph[src_id][edge_pos].weight;
            }
            current_edge += supported_vector_length;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void VectorisedCSRGraph<_TVertexValue, _TEdgeWeight>::calculate_incoming_sizes()
{
    int vertices_count = this->vertices_count;
    for(int i = 0; i < vertices_count; i++)
    {
        incoming_sizes_per_vertex[i] = 0;
    }
    
    // process first part
    for(int src_id = 0; src_id < number_of_vertices_in_first_part; src_id++)
    {
        long long edge_start = first_part_ptrs[src_id];
        int connections_count = first_part_sizes[src_id];
        
        for(int edge_pos = 0; edge_pos < connections_count; edge_pos++)
        {
            int dst_id = outgoing_ids[edge_start + edge_pos];
            incoming_sizes_per_vertex[dst_id]++;
        }
    }
    
    // process last part
    for(int cur_vector_segment = 0; cur_vector_segment < vector_segments_count; cur_vector_segment++)
    {
        int segment_first_vertex = cur_vector_segment * supported_vector_length + number_of_vertices_in_first_part;
        long long segement_edges_start = vector_group_ptrs[cur_vector_segment];
        int segment_connections_count  = vector_group_sizes[cur_vector_segment];
        
        for(int edge_pos = 0; edge_pos < segment_connections_count; edge_pos++)
        {
            for(int i = 0; i < supported_vector_length; i++)
            {
                int src_id = segment_first_vertex + i;
                int dst_id = outgoing_ids[segement_edges_start + edge_pos * supported_vector_length + i];
                incoming_sizes_per_vertex[dst_id]++;
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void VectorisedCSRGraph<_TVertexValue, _TEdgeWeight>::import_graph(EdgesListGraph<_TVertexValue, _TEdgeWeight>
                                                                   &_old_graph,
                                                                   VerticesState _vertices_state,
                                                                   EdgesState _edges_state,
                                                                   int _supported_vector_length,
                                                                   SupportedTraversalType _traversal_type,
                                                                   bool _free_initial_graph)
{
    // set optimisation parameters
    this->vertices_state          = _vertices_state;
    this->edges_state             = _edges_state;
    this->supported_vector_length = _supported_vector_length;
    long long old_edges_count     = _old_graph.get_edges_count();
    
    // create tmp graph
    int tmp_vertices_count = _old_graph.get_vertices_count();
    long long tmp_edges_count = _old_graph.get_edges_count();
    vector<vector<TempEdgeData<_TEdgeWeight> > > tmp_graph;
    vector<_TVertexValue> tmp_vertex_data;
    
    construct_tmp_graph_from_edges_list(_old_graph, tmp_graph, tmp_vertex_data, tmp_vertices_count, _traversal_type);
    
    if(_free_initial_graph)
        _old_graph.~EdgesListGraph<_TVertexValue, _TEdgeWeight>(); // free memory of initial graph
    
    // sort vertices by outgoing degree size
    int *tmp_reordered_vertex_ids = new int[tmp_vertices_count];
    sort_vertices_in_descending_order(tmp_graph, tmp_reordered_vertex_ids, tmp_vertices_count);
    
    // sort adjacent ids locally for each vertex
    sort_edges(tmp_graph, tmp_vertices_count);
    
    // need to do something here....
    int threshold_vertex = calculate_and_find_threshold_vertex(tmp_graph, tmp_vertices_count);
    
    // flatten graph for vectorisation
    flatten_graph(tmp_graph, tmp_vertices_count, tmp_edges_count, old_edges_count, threshold_vertex);
    
    // get new pointers
    this->resize(tmp_vertices_count, tmp_edges_count, threshold_vertex);
    
    // save optimised graph
    convert_tmp_graph_into_vect_CSR(tmp_graph, tmp_reordered_vertex_ids, tmp_vertex_data, tmp_vertices_count);
    
    // calucalte incoming sizes
    calculate_incoming_sizes();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* init_graph_helpers_hpp */
