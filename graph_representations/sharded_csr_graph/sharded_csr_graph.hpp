#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ShardedCSRGraph::ShardedCSRGraph()
{
    this->graph_type = SHARDED_CSR_GRAPH;
    max_cached_vertices = 1;
    shards_number = 0;
    outgoing_shards = NULL;
    incoming_shards = NULL;

    vertices_reorder_buffer = NULL;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ShardedCSRGraph::~ShardedCSRGraph()
{
    MemoryAPI::free_array(outgoing_shards);
    MemoryAPI::free_array(incoming_shards);
    MemoryAPI::free_array(vertices_reorder_buffer);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
void process()
{
    Frontier frontier; // need to process verices {1, 2, 3, 10, 15}
    // problem: each shard can have different enumeration...
    // если сначала сортируем, потом фиксируем номера внутри каждой shard в таком же порядке
    // - может происходить разброс степеней соседних
    //

    // если сортируем каждую шарду - разная нумерация, как работать с фронтом?
    // можно смотреть на тип фронта
    // all-active - нет проблемы
    // dense - нет проблемы
    // если есть sparse-часть - её перенумеровывать для каждой шарды (внутри advance)
*/

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*void ShardedCSRGraph::import(EdgesListGraph &_old_graph,
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
}*/

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
