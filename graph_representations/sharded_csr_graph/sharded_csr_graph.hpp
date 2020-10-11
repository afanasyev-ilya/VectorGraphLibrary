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

void ShardedGraph::import_graph(EdgesListGraph &_el_graph)
{
    cout << "sharded import" << endl;
    this->vertices_count = _el_graph.get_vertices_count();
    this->edges_count = _el_graph.get_edges_count();

    max_cached_vertices = 1*1024*1024/(sizeof(int));
    cout << "max_cached_vertices: " << max_cached_vertices << endl;

    shards_number = (this->vertices_count - 1)/max_cached_vertices + 1;
    cout << "shards number" << shards_number << endl;

    outgoing_shards = new VectCSRGraph[shards_number];

    _el_graph.transpose();
    _el_graph.preprocess_into_csr_based();
    _el_graph.transpose();

    for(int i = 0; i < 20; i++)
        cout << _el_graph.get_src_ids()[i] << " " << _el_graph.get_dst_ids()[i] << endl;

    // estimate edge borders in each shards
    int edges_in_each_shard[shards_number]; // TODO dynamic
    int front_border[shards_number]; // TODO dynamic
    int back_border[shards_number]; // TODO dynamic
    long long first_shard_edge[shards_number];
    long long second_shard_edge[shards_number];
    for(int i = 0; i < shards_number; i++)
    {
        edges_in_each_shard[i] = 0;
        first_shard_edge[i] = 0;
        second_shard_edge[i] = 0;
        front_border[i] = max_cached_vertices * i;
        back_border[i] = max_cached_vertices * (i + 1);
    }

    for(int i = 0; i < shards_number; i++)
    {
        cout << "[" << front_border[i] << ", " << back_border[i] << "]" << endl;
    }

    int *dst_ids = _el_graph.get_dst_ids(); // since transposed all dst ids are sorted
    for(long long i = 0; i < (this->edges_count - 1); i++)
    {
        int dst_id = dst_ids[i];
        int next_dst_id = dst_ids[i + 1];
        int shard_id = get_shard_id(dst_id);
        edges_in_each_shard[shard_id]++;
    }

    for(int i = 0; i < shards_number; i++)
        cout << edges_in_each_shard[i] << " edges in shard " << i << ", " << 100.0*((double)edges_in_each_shard[i])/this->edges_count << " %" << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/*void ShardedGraph::import_graph(EdgesListGraph &_old_graph,
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
