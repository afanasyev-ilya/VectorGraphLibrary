//
//  shard.hpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 09/08/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef shard_vect_CSR_hpp
#define shard_vect_CSR_hpp

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TEdgeWeight>
ShardVectCSR<_TEdgeWeight>::ShardVectCSR()
{
    this->vertices_in_shard = 0;
    this->edges_in_shard = 0;
    
    this->global_src_ids = NULL;
    vector_group_ptrs = NULL;
    vector_group_sizes = NULL;
    dst_ids = NULL;
    weights  = NULL;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TEdgeWeight>
ShardVectCSR<_TEdgeWeight>::~ShardVectCSR()
{
    clear();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TEdgeWeight>
void ShardVectCSR<_TEdgeWeight>::clear()
{
    this->vertices_in_shard = 0;
    this->edges_in_shard = 0;
    
    if(vector_group_ptrs != NULL)
        delete[] vector_group_ptrs;
    vector_group_ptrs = NULL;

    if(vector_group_sizes != NULL)
        delete[] vector_group_sizes;
    vector_group_sizes = NULL;
    
    if(dst_ids != NULL)
        delete[] dst_ids;
    dst_ids = NULL;

    if(weights != NULL)
        delete[] weights;
    weights = NULL;
    
    if(this->global_src_ids != NULL)
        delete[] this->global_src_ids;
    this->global_src_ids = NULL;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TEdgeWeight>
void ShardVectCSR<_TEdgeWeight>::resize(int _vertices_in_shard, long long _edges_in_shard)
{
    clear();
    this->vertices_in_shard = _vertices_in_shard;
    this->edges_in_shard = _edges_in_shard;
    
    int number_of_vector_groups = (this->vertices_in_shard - 1) / VECTOR_LENGTH_IN_SHARD + 1;
    vector_group_ptrs = new long long[number_of_vector_groups];
    vector_group_sizes = new int[number_of_vector_groups];
    dst_ids     = new int[this->edges_in_shard];
    weights     = new _TEdgeWeight[this->edges_in_shard];
    this->global_src_ids = new int[this->vertices_in_shard];
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TEdgeWeight>
void ShardVectCSR<_TEdgeWeight>::init_shard_from_tmp_map()
{
    // edges_in_shard and vertices_in_shard are already set from add_edge
    int number_of_vector_groups = (this->vertices_in_shard - 1) / VECTOR_LENGTH_IN_SHARD + 1;
    vector_group_ptrs = new long long[number_of_vector_groups];
    vector_group_sizes = new int[number_of_vector_groups];
    this->global_src_ids = new int[this->vertices_in_shard];
    
    for(int i = 0; i < number_of_vector_groups; i++)
        vector_group_sizes[i] = 0;
    
    int src_id = 0;
    vector<pair<int, int> > pairs(this->vertices_in_shard);
    vector<vector<TmpMapEdge <_TEdgeWeight> > > tmp_graph(this->vertices_in_shard);
    
    // fill tmp graph from map
    typedef typename map< int,vector< TmpMapEdge<_TEdgeWeight> > >::iterator map_iterator;
    vector<int> old_global_ids(this->vertices_in_shard);
    for(map_iterator it = this->tmp_map_data.begin(); it != this->tmp_map_data.end(); it++)
    {
        int global_src_id = it->first;
        vector< TmpMapEdge<_TEdgeWeight> > &adj_list_ptr = it->second;
        for(int i = 0; i < adj_list_ptr.size(); i++)
            tmp_graph[src_id].push_back(adj_list_ptr[i]);
        pairs[src_id] = make_pair(tmp_graph[src_id].size(), src_id);
        old_global_ids[src_id] = it->first;
        src_id++;
    }
    
    this->tmp_map_data.clear();
    
    // sort all vertices now
    sort(pairs.begin(), pairs.end());
    reverse(pairs.begin(), pairs.end());
    
    // save old indexes array
    int *old_indexes = new int[this->vertices_in_shard];
    for(int i = 0; i < this->vertices_in_shard; i++)
    {
        old_indexes[i] = pairs[i].second;
    }
    
    // sort graph now (in 2 steps)
    vector<vector<TmpMapEdge<_TEdgeWeight> > > new_tmp_graph(this->vertices_in_shard);
    for(int i = 0; i < this->vertices_in_shard; i++)
    {
        new_tmp_graph[i] = tmp_graph[old_indexes[i]];
    }
    
    for(int i = 0; i < this->vertices_in_shard; i++)
    {
        tmp_graph[i] = new_tmp_graph[i];
    }
    
    new_tmp_graph.clear();
    delete []old_indexes;
    
    // calculate new vector sizes
    for(int i = 0; i < this->vertices_in_shard; i++)
    {
        int global_src_id = old_global_ids[pairs[i].second];
        vector< TmpMapEdge<_TEdgeWeight> > &adj_list_ptr = tmp_graph[i];
        
        int current_vector_group_pos = i / VECTOR_LENGTH_IN_SHARD;
        
        if(adj_list_ptr.size() > vector_group_sizes[current_vector_group_pos])
        {
            vector_group_sizes[current_vector_group_pos] = adj_list_ptr.size();
        }
    }
    
    // append map with edges
    long long check_edges = 0;
    int additional_edges = 0;
    int last_size = 0;
    for(int i = 0; i < this->vertices_in_shard; i++)
    {
        int global_src_id = old_global_ids[pairs[i].second];
        vector< TmpMapEdge<_TEdgeWeight> > &adj_list_ptr = tmp_graph[i];
        
        int current_vector_group_pos = i / VECTOR_LENGTH_IN_SHARD;
        
        if((i % VECTOR_LENGTH_IN_SHARD) == 0)
        {
            last_size = vector_group_sizes[current_vector_group_pos];
        }
        
        while(adj_list_ptr.size() != vector_group_sizes[current_vector_group_pos])
        {
            TmpMapEdge<_TEdgeWeight> tmp_data;
            tmp_data.dst_id = global_src_id;
            tmp_data.weight = 0.0;
            adj_list_ptr.push_back(tmp_data);
            additional_edges++;
        }
    }
    
    this->edges_in_shard += additional_edges + VECTOR_LENGTH_IN_SHARD*last_size;
    dst_ids = new int[this->edges_in_shard];
    weights = new _TEdgeWeight[this->edges_in_shard];
    
    // convert
    long long edge_pos = 0;
    for(int src_id = 0; src_id < this->vertices_in_shard; src_id++)
    {
        int current_vector_group_pos = src_id / VECTOR_LENGTH_IN_SHARD;
        int connections_count = vector_group_sizes[current_vector_group_pos];
        
        int global_src_id = old_global_ids[pairs[src_id].second];
        vector< TmpMapEdge<_TEdgeWeight> > &adj_list_ptr = tmp_graph[src_id];
        this->global_src_ids[src_id] = global_src_id;
        
        if((src_id % VECTOR_LENGTH_IN_SHARD) == 0)
            vector_group_ptrs[current_vector_group_pos] = edge_pos;
        
        for(int i = 0; i < connections_count; i++)
        {
            TmpMapEdge<_TEdgeWeight> cur_edge = adj_list_ptr[i];
            dst_ids[edge_pos] = cur_edge.dst_id;
            weights[edge_pos] = cur_edge.weight;

            edge_pos += VECTOR_LENGTH_IN_SHARD;
        }
        
        if((src_id % VECTOR_LENGTH_IN_SHARD) != (VECTOR_LENGTH_IN_SHARD - 1))
        {
            edge_pos -= VECTOR_LENGTH_IN_SHARD * connections_count;
        }
        else
        {
            edge_pos -= VECTOR_LENGTH_IN_SHARD;
        }
        edge_pos++;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TEdgeWeight>
void ShardVectCSR<_TEdgeWeight>::print()
{
    cout << "printing" << endl;
    int vector_segments_count = (this->vertices_in_shard - 1) / VECTOR_LENGTH_IN_SHARD + 1;
    
    for(int cur_vector_segment = 0; cur_vector_segment < vector_segments_count; cur_vector_segment++)
    {
        int vec_start = cur_vector_segment * VECTOR_LENGTH_IN_SHARD;
        long long edge_start = vector_group_ptrs[cur_vector_segment];
        int cur_max_connections_count = vector_group_sizes[cur_vector_segment];
        
        for(int edge_pos = 0; edge_pos < cur_max_connections_count; edge_pos++)
        {
            cout << "(";
            for(int i = 0; i < VECTOR_LENGTH_IN_SHARD; i++)
            {
                int cur_vertex = vec_start + i;
                if(cur_vertex < this->vertices_in_shard)
                {
                    cout << dst_ids[edge_start + edge_pos * VECTOR_LENGTH_IN_SHARD + i] << ", ";
                }
            }
            cout << ")" << endl;
            cout << edge_pos << endl;
        }
        cout << endl;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TEdgeWeight>
ShardVectCSRPointerData<_TEdgeWeight> ShardVectCSR<_TEdgeWeight>::get_pointers_data()
{
    ShardVectCSRPointerData<_TEdgeWeight> pointers;
    
    pointers.vertices_in_shard = this->vertices_in_shard;
    pointers.vector_group_ptrs = vector_group_ptrs;
    pointers.vector_group_sizes = vector_group_sizes;
    pointers.dst_ids = dst_ids;
    pointers.weights = weights;
    pointers.global_src_ids = this->global_src_ids;
    
    return pointers;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _TEdgeWeight>
void ShardVectCSR<_TEdgeWeight>::move_to_device()
{
    throw "move to device not implemented yet";
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _TEdgeWeight>
void ShardVectCSR<_TEdgeWeight>::move_to_host()
{
    throw "move to host not implemented yet";
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TEdgeWeight>
void ShardVectCSR<_TEdgeWeight>::save_to_binary_file(FILE *_graph_file)
{
    int vertices_in_shard = this->vertices_in_shard;
    long long edges_in_shard = this->edges_in_shard;
    fwrite(reinterpret_cast<const void*>(&vertices_in_shard), sizeof(int), 1, _graph_file);
    fwrite(reinterpret_cast<const void*>(&edges_in_shard), sizeof(long long), 1, _graph_file);
    
    int number_of_vector_groups = (this->vertices_in_shard - 1) / VECTOR_LENGTH_IN_SHARD + 1;
    fwrite(reinterpret_cast<const char*>(this->global_src_ids), sizeof(int), vertices_in_shard, _graph_file);
    fwrite(reinterpret_cast<const char*>(vector_group_ptrs), sizeof(long long), number_of_vector_groups, _graph_file);
    fwrite(reinterpret_cast<const char*>(vector_group_sizes), sizeof(int), number_of_vector_groups, _graph_file);
    fwrite(reinterpret_cast<const char*>(dst_ids), sizeof(int), edges_in_shard, _graph_file);
    fwrite(reinterpret_cast<const char*>(weights), sizeof(_TEdgeWeight), edges_in_shard, _graph_file);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TEdgeWeight>
void ShardVectCSR<_TEdgeWeight>::load_from_binary_file(FILE *_graph_file)
{
    int vertices_in_shard = 0;
    long long edges_in_shard = 0;
    
    fread(reinterpret_cast<void*>(&vertices_in_shard), sizeof(int), 1, _graph_file);
    fread(reinterpret_cast<void*>(&edges_in_shard), sizeof(long long), 1, _graph_file);
    
    this->resize(vertices_in_shard, edges_in_shard);
    
    int number_of_vector_groups = (this->vertices_in_shard - 1) / VECTOR_LENGTH_IN_SHARD + 1;
    fread(reinterpret_cast<char*>(this->global_src_ids), sizeof(int), vertices_in_shard, _graph_file);
    fread(reinterpret_cast<char*>(vector_group_ptrs), sizeof(long long), number_of_vector_groups, _graph_file);
    fread(reinterpret_cast<char*>(vector_group_sizes), sizeof(int), number_of_vector_groups, _graph_file);
    fread(reinterpret_cast<char*>(dst_ids), sizeof(int), edges_in_shard, _graph_file);
    fread(reinterpret_cast<char*>(weights), sizeof(_TEdgeWeight), edges_in_shard, _graph_file);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* shard_hp */
