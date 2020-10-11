#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ShardedGraph::ShardedGraph()
{
    this->graph_type = SHARDED_CSR_GRAPH;
    max_cached_vertices = 1;
    shards_number = 0;
    outgoing_shards = NULL;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ShardedGraph::~ShardedGraph()
{
    if(outgoing_shards != NULL)
        delete []outgoing_shards;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ShardedGraph::import(EdgesListGraph &_el_graph)
{
    this->vertices_count = _el_graph.get_vertices_count();
    this->edges_count = _el_graph.get_edges_count();

    max_cached_vertices = this->vertices_count/2; // 1*1024*1024/(sizeof(int)); TODO
    cout << "max_cached_vertices: " << max_cached_vertices << endl;

    shards_number = (this->vertices_count - 1)/max_cached_vertices + 1;
    cout << "shards number: " << shards_number << endl;

    outgoing_shards = new UndirectedCSRGraph[shards_number];

    _el_graph.transpose();
    _el_graph.preprocess_into_csr_based();
    _el_graph.transpose();


    // obtain pointers, edges inside sorted according to dst_ids
    int *el_src_ids = _el_graph.get_src_ids();
    int *el_dst_ids = _el_graph.get_dst_ids();

    // estimate edge borders in each shards
    int first_border[shards_number]; // TODO dynamic
    int last_border[shards_number];
    long long first_shard_edge[shards_number];
    long long last_shard_edge[shards_number];
    for(int i = 0; i < shards_number; i++)
    {
        first_shard_edge[i] = 0;
        last_shard_edge[i] = 0;
        first_border[i] = max_cached_vertices * i;
        last_border[i] = max_cached_vertices * (i + 1);
    }

    Timer tm;
    tm.start();
    first_shard_edge[0] = 0;
    #pragma _NEC ivdep
    #pragma omp parallel for
    for(long long edge_pos = 0; edge_pos < (this->edges_count - 1); edge_pos++)
    {
        int dst_id = el_dst_ids[edge_pos];
        int next_dst_id = el_dst_ids[edge_pos + 1];

        #pragma unroll
        for(int shard_id = 0; shard_id < shards_number; shard_id++)
        {
            int first_border_val = first_border[shard_id];
            int last_border_val = last_border[shard_id];
            if((dst_id < first_border_val) && (next_dst_id >= first_border_val))
            {
                first_shard_edge[shard_id] = edge_pos;
            }
            if((dst_id < last_border_val) && (next_dst_id >= last_border_val))
            {
                last_shard_edge[shard_id] = edge_pos;
            }
        }
    }
    last_shard_edge[shards_number - 1] = this->edges_count;
    tm.end();
    tm.print_bandwidth_stats("Sharded split", this->edges_count, sizeof(int)*2.0);

    for(int shard_id = 0; shard_id < shards_number; shard_id++)
    {
        long long shard_size = last_shard_edge[shard_id] - first_shard_edge[shard_id];
        cout << "[" << first_shard_edge[shard_id] << ", " << last_shard_edge[shard_id] << "], size = " << shard_size << " | " << 100.0*(shard_size)/this->edges_count << " %" << endl;
    }

    // import each shard from edges list
    for(int shard_id = 0; shard_id < shards_number; shard_id++)
    {
        cout << " ------------------------------------------------------------ " << endl;
        cout << "shard " << shard_id << endl;
        long long edges_in_shard = last_shard_edge[shard_id] - first_shard_edge[shard_id];

        long long first_shard_edge_val = first_shard_edge[shard_id];

        int *shard_src_ids_ptr = &el_src_ids[first_shard_edge_val];
        int *shard_dst_ids_ptr = &el_dst_ids[first_shard_edge_val];
        EdgesListGraph edges_list_shard;
        edges_list_shard.import(shard_src_ids_ptr, shard_dst_ids_ptr, this->vertices_count, edges_in_shard);

        outgoing_shards[shard_id].import(edges_list_shard, NULL);
        outgoing_shards[shard_id].print();
        cout << " ------------------------------------------------------------ " << endl;
    }
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


/*void ShardedGraph::import(EdgesListGraph &_old_graph,
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
