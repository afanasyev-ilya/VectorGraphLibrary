//
//  sharded_graph.hpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 09/08/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef sharded_graph_hpp
#define sharded_graph_hpp

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


ShardedGraph::ShardedGraph(ShardType _type_of_shard, int _cache_size)
{
    shards_data = NULL;
    type_of_shard = _type_of_shard;
    cache_size = _cache_size;
    max_cached_vertices = cache_size / sizeof(int);
    //max_cached_vertices = 1024;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


ShardedGraph::~ShardedGraph()
{
    clear();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void ShardedGraph::clear()
{
    if(shards_data != NULL)
    {
        for(int i = 0; i < number_of_shards; i++)
        {
            shards_data[i]->clear();
        }
        delete []shards_data;
    }
    
    shards_data = NULL;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void ShardedGraph::import_graph(EdgesListGraph &_old_graph,
                                                             AlgorithmTraversalType _traversal_type)
{
    LOAD_EDGES_LIST_GRAPH_DATA(_old_graph);
    this->vertices_count = vertices_count;
    this->edges_count    = edges_count;
    
    number_of_shards = (vertices_count - 1) / max_cached_vertices + 1;
    
    shards_data = new ShardBase<_TEdgeWeight>*[number_of_shards];
    for(int i = 0; i < number_of_shards; i++)
    {
        if(type_of_shard == SHARD_CSR_TYPE)
            shards_data[i] = new ShardCSR<_TEdgeWeight>;
        else if(type_of_shard == SHARD_VECT_CSR_TYPE)
            shards_data[i] = new ShardVectCSR<_TEdgeWeight>;
    }
    
    int total_threads_num = omp_get_max_threads();
    
    double t1 = omp_get_wtime();
    #pragma omp parallel num_threads(total_threads_num)
    {
        int tid = omp_get_thread_num();
        
        for(long long i = 0; i < this->edges_count; i++)
        {
            int src_id, dst_id;
            if(_traversal_type == PUSH_TRAVERSAL)
            {
                src_id = src_ids[i];
                dst_id = dst_ids[i];
            }
            else if(_traversal_type == PULL_TRAVERSAL)
            {
                dst_id = src_ids[i];
                src_id = dst_ids[i];
            }
            int shard_id = this->get_shard_id(dst_id);
            
            if((shard_id % total_threads_num) == tid)
            {
                _TEdgeWeight weight = weights[i];
                shards_data[shard_id]->add_edge_to_tmp_map(src_id, dst_id, weight);
            }
        }
    }
    double t2 = omp_get_wtime();
    cout << "init map time: " << t2 - t1 << " sec" << endl;
    
    t1 = omp_get_wtime();
    #pragma omp parallel for num_threads(total_threads_num)
    for(int i = 0; i < number_of_shards; i++)
    {
        shards_data[i]->init_shard_from_tmp_map();
    }
    t2 = omp_get_wtime();
    cout << "convert from map to shard time: " << t2 - t1 << " sec" << endl;
    
    cout << "old edges count: " << this->edges_count << endl;
    long long new_edges_count = 0;
    for(int i = 0; i < number_of_shards; i++)
    {
        new_edges_count += shards_data[i]->get_edges_in_shard();
    }
    cout << "new edges count: " << new_edges_count << endl;
    this->edges_count = new_edges_count;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void ShardedGraph::print()
{
    cout << "vertices in each shard: " << max_cached_vertices << endl;
    cout << "shards number: " << number_of_shards << endl;
    for(int i = 0; i < number_of_shards; i++)
    {
        shards_data[i]->print();
    }
    cout << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void ShardedGraph::print_stats()
{
    cout << "NUMBER OF SHARDS: " << number_of_shards << endl;
    for(int i = 0; i < number_of_shards; i++)
    {
        cout << "SHARD № " << i+1 << endl;
        shards_data[i]->print_stats(this->vertices_count, this->edges_count);
        cout << endl;
    }
    
    double total_edges = 0;
    double total_vertices = 0;
    for(int i = 0; i < number_of_shards; i++)
    {
        total_vertices += shards_data[i]->get_vertices_in_shard();
        total_edges += shards_data[i]->get_edges_in_shard();
    }
    
    cout << "avg vertices in each shard: " << 100.0 * (total_vertices / number_of_shards) / this->vertices_count << " %" << endl;
    cout << "avg edges in each shard: " << 100.0 * (total_edges / number_of_shards) / this->edges_count << " %" << endl;
    cout << "edges_count: " << this->edges_count << endl;
    double edges_count = total_edges;
    double t_k = 4.8;
    double segments_count = number_of_shards;
    double avg_vertices_in_segment = total_vertices / number_of_shards;
    cout << "estimated acceleration = " << (edges_count * t_k) / (2.0 * segments_count * avg_vertices_in_segment + edges_count) << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template<typename _T>
_T *ShardedGraph::allocate_local_shard_data()
{
    int max_vertices_in_shard = 0;
    for(int i = 0; i < number_of_shards; i++)
    {
        if(max_vertices_in_shard < shards_data[i]->get_vertices_in_shard())
            max_vertices_in_shard = shards_data[i]->get_vertices_in_shard();
    }
    
    _T *local_data = new _T[max_vertices_in_shard];
    return local_data;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__

void ShardedGraph::move_to_device()
{
    for(int i = 0; i < number_of_shards; i++)
    {
        shards_data[i]->move_to_device();
    }
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__

void ShardedGraph::move_to_host()
{
    for(int i = 0; i < number_of_shards; i++)
    {
        shards_data[i]->move_to_host();
    }
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


bool ShardedGraph::save_to_binary_file(string _file_name)
{
    // write header
    FILE * graph_file = fopen(_file_name.c_str(), "wb");
    if(graph_file == NULL)
        return false;
    
    // save header
    fwrite(reinterpret_cast<const void*>(&(this->vertices_count)), sizeof(int), 1, graph_file);
    fwrite(reinterpret_cast<const void*>(&(this->edges_count)), sizeof(long long), 1, graph_file);
    fwrite(reinterpret_cast<const void*>(&cache_size), sizeof(int), 1, graph_file);
    fwrite(reinterpret_cast<const void*>(&max_cached_vertices), sizeof(int), 1, graph_file);
    fwrite(reinterpret_cast<const void*>(&number_of_shards), sizeof(int), 1, graph_file);
    fwrite(reinterpret_cast<const void*>(&type_of_shard), sizeof(ShardType), 1, graph_file);
    
    // save shard data
    for(int i = 0; i < number_of_shards; i++)
    {
        shards_data[i]->save_to_binary_file(graph_file);
    }
    
    fclose(graph_file);
    
    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


bool ShardedGraph::load_from_binary_file(string _file_name)
{
    FILE * graph_file = fopen(_file_name.c_str(), "rb");
    if(graph_file == NULL)
        return false;
    
    // clear existing data
    this->clear();
    
    // read header
    fread(reinterpret_cast<void*>(&(this->vertices_count)), sizeof(int), 1, graph_file);
    fread(reinterpret_cast<void*>(&(this->edges_count)), sizeof(long long), 1, graph_file);
    fread(reinterpret_cast<void*>(&cache_size), sizeof(int), 1, graph_file);
    fread(reinterpret_cast<void*>(&max_cached_vertices), sizeof(int), 1, graph_file);
    fread(reinterpret_cast<void*>(&number_of_shards), sizeof(int), 1, graph_file);
    fread(reinterpret_cast<void*>(&type_of_shard), sizeof(ShardType), 1, graph_file);
    
    // load shards data
    shards_data = new ShardBase<_TEdgeWeight>*[number_of_shards];
    for(int i = 0; i < number_of_shards; i++)
    {
        if(type_of_shard == SHARD_CSR_TYPE)
            shards_data[i] = new ShardCSR<_TEdgeWeight>;
        else if(type_of_shard == SHARD_VECT_CSR_TYPE)
            shards_data[i] = new ShardVectCSR<_TEdgeWeight>;
        shards_data[i]->load_from_binary_file(graph_file);
    }
    
    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* sharded_graph_hpp */
