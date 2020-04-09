//
//  shard.hpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 09/08/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef shard_CSR_hpp
#define shard_CSR_hpp

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TEdgeWeight>
ShardCSR<_TEdgeWeight>::ShardCSR()
{
    this->vertices_in_shard = 0;
    this->edges_in_shard = 0;
    
    vertex_ptrs = NULL;
    dst_ids = NULL;
    weights = NULL;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TEdgeWeight>
ShardCSR<_TEdgeWeight>::~ShardCSR()
{
    clear();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TEdgeWeight>
void ShardCSR<_TEdgeWeight>::clear()
{
    this->vertices_in_shard = 0;
    this->edges_in_shard = 0;
    
    if(vertex_ptrs != NULL)
        delete[] vertex_ptrs;
    if(dst_ids != NULL)
        delete[] dst_ids;
    if(weights != NULL)
        delete[] weights;
    if(this->global_src_ids != NULL)
        delete[] this->global_src_ids;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TEdgeWeight>
void ShardCSR<_TEdgeWeight>::resize(int _vertices_in_shard, long long _edges_in_shard)
{
    clear();
    this->vertices_in_shard = _vertices_in_shard;
    this->edges_in_shard = _edges_in_shard;
    
    vertex_ptrs = new long long[this->vertices_in_shard + 1];
    dst_ids     = new int[this->edges_in_shard];
    weights     = new _TEdgeWeight[this->edges_in_shard];
    this->global_src_ids = new int[this->vertices_in_shard];
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TEdgeWeight>
void ShardCSR<_TEdgeWeight>::init_shard_from_tmp_map()
{
    // edges_in_shard and vertices_in_shard are already set from add_edge
    vertex_ptrs = new long long[this->vertices_in_shard + 1];
    dst_ids     = new int[this->edges_in_shard];
    weights     = new _TEdgeWeight[this->edges_in_shard];
    this->global_src_ids = new int[this->vertices_in_shard];
    
    int vertex_pos = 0;
    long long edge_pos = 0;
    typedef typename map< int,vector< TmpMapEdge<_TEdgeWeight> > >::iterator map_iterator;
    for(map_iterator it = this->tmp_map_data.begin(); it != this->tmp_map_data.end(); ++it)
    {
        int global_src_id = it->first;
        vector< TmpMapEdge<_TEdgeWeight> > &adj_list_ptr = it->second;
        
        vertex_ptrs[vertex_pos] = edge_pos;
        this->global_src_ids[vertex_pos] = global_src_id;
        
        for(int i = 0; i < adj_list_ptr.size(); i++)
        {
            TmpMapEdge<_TEdgeWeight> cur_edge = adj_list_ptr[i];
            dst_ids[edge_pos] = cur_edge.dst_id;
            weights[edge_pos] = cur_edge.weight;
            edge_pos++;
        }
        vertex_pos++;
    }
    vertex_ptrs[this->vertices_in_shard] = edge_pos;
    
    this->tmp_map_data.clear();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TEdgeWeight>
void ShardCSR<_TEdgeWeight>::print()
{
    for(int vertex_pos = 0; vertex_pos < this->vertices_in_shard; vertex_pos++)
    {
        cout << "vertex " << vertex_pos << " with global id " << this->global_src_ids[vertex_pos] << " connected to: ";
        long long edge_pos = vertex_ptrs[vertex_pos];
        for(long long edge_pos = vertex_ptrs[vertex_pos]; edge_pos < vertex_ptrs[vertex_pos + 1]; edge_pos++)
            cout << "(" << dst_ids[edge_pos] << "," << weights[edge_pos] << ")" << " ";
        cout << endl;
    }
    cout << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TEdgeWeight>
ShardCSRPointerData<_TEdgeWeight> ShardCSR<_TEdgeWeight>::get_pointers_data()
{
    ShardCSRPointerData<_TEdgeWeight> pointers;
    pointers.vertex_ptrs = vertex_ptrs;
    pointers.dst_ids = dst_ids;
    pointers.weights = weights;
    pointers.global_src_ids = this->global_src_ids;
    pointers.vertices_in_shard = this->vertices_in_shard;
    return pointers;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _TEdgeWeight>
void ShardCSR<_TEdgeWeight>::move_to_device()
{
    MemoryAPI::move_array_to_device<long long>(&vertex_ptrs, this->vertices_in_shard + 1);
    MemoryAPI::move_array_to_device<int>(&dst_ids, this->edges_in_shard);
    MemoryAPI::move_array_to_device<_TEdgeWeight>(&weights, this->edges_in_shard);
    MemoryAPI::move_array_to_device<int>(&(this->global_src_ids), this->vertices_in_shard);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _TEdgeWeight>
void ShardCSR<_TEdgeWeight>::move_to_host()
{
    MemoryAPI::move_array_to_host<long long>(&vertex_ptrs, this->vertices_in_shard + 1);
    MemoryAPI::move_array_to_host<int>(&dst_ids, this->edges_in_shard);
    MemoryAPI::move_array_to_host<_TEdgeWeight>(&weights, this->edges_in_shard);
    MemoryAPI::move_array_to_host<int>(&(this->global_src_ids), this->vertices_in_shard);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TEdgeWeight>
void ShardCSR<_TEdgeWeight>::save_to_binary_file(FILE *_graph_file)
{
    int vertices_in_shard = this->vertices_in_shard;
    long long edges_in_shard = this->edges_in_shard;
    fwrite(reinterpret_cast<const void*>(&vertices_in_shard), sizeof(int), 1, _graph_file);
    fwrite(reinterpret_cast<const void*>(&edges_in_shard), sizeof(long long), 1, _graph_file);
    
    fwrite(reinterpret_cast<const char*>(this->global_src_ids), sizeof(int), vertices_in_shard, _graph_file);
    fwrite(reinterpret_cast<const char*>(vertex_ptrs), sizeof(long long), vertices_in_shard + 1, _graph_file);
    fwrite(reinterpret_cast<const char*>(dst_ids), sizeof(int), edges_in_shard, _graph_file);
    fwrite(reinterpret_cast<const char*>(weights), sizeof(_TEdgeWeight), edges_in_shard, _graph_file);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TEdgeWeight>
void ShardCSR<_TEdgeWeight>::load_from_binary_file(FILE *_graph_file)
{
    int vertices_in_shard = 0;
    long long edges_in_shard = 0;
    
    fread(reinterpret_cast<void*>(&vertices_in_shard), sizeof(int), 1, _graph_file);
    fread(reinterpret_cast<void*>(&edges_in_shard), sizeof(long long), 1, _graph_file);
    
    this->resize(vertices_in_shard, edges_in_shard);
    
    fread(reinterpret_cast<char*>(this->global_src_ids), sizeof(int), vertices_in_shard, _graph_file);
    fread(reinterpret_cast<char*>(vertex_ptrs), sizeof(long long), vertices_in_shard + 1, _graph_file);
    fread(reinterpret_cast<char*>(dst_ids), sizeof(int), edges_in_shard, _graph_file);
    fread(reinterpret_cast<char*>(weights), sizeof(_TEdgeWeight), edges_in_shard, _graph_file);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* shard_hp */
