#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ShardedCSRGraph::import_direction(EdgesListGraph &_el_graph, TraversalDirection _import_direction)
{
    _el_graph.transpose();
    _el_graph.preprocess_into_csr_based();
    _el_graph.transpose(); // dst ids are sorted here

    // obtain pointers, edges inside sorted according to dst_ids
    int *el_src_ids = _el_graph.get_src_ids();
    int *el_dst_ids = _el_graph.get_dst_ids();

    // estimate edge borders in each shards
    int *first_border = new int[shards_number];
    int *last_border = new int[shards_number];
    long long *first_shard_edge = new long long[shards_number];
    long long *last_shard_edge = new long long[shards_number];
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

        //#pragma unroll
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
    //#ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_time_stats("Sharded split");
    //#endif

    // import each shard from edges list
    tm.start();
    for(int shard_id = 0; shard_id < shards_number; shard_id++)
    {
        long long edges_in_shard = last_shard_edge[shard_id] - first_shard_edge[shard_id];

        long long first_shard_edge_val = first_shard_edge[shard_id];

        int *shard_src_ids_ptr = &(el_src_ids[first_shard_edge_val]);
        int *shard_dst_ids_ptr = &(el_dst_ids[first_shard_edge_val]);
        cout << shard_id << " : " << edges_in_shard << " " << 1.0*edges_in_shard/_el_graph.get_edges_count() << endl;
        EdgesListGraph edges_list_shard;
        edges_list_shard.import(shard_src_ids_ptr, shard_dst_ids_ptr, this->vertices_count, edges_in_shard);

        if(_import_direction == SCATTER)
            outgoing_shards[shard_id].import(edges_list_shard, NULL);
        else if(_import_direction == GATHER)
            incoming_shards[shard_id].import(edges_list_shard, NULL);
    }
    tm.end();
    //#ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_time_stats("Import shards");
    //#endif

    delete []first_border;
    delete []last_border;
    delete []first_shard_edge;
    delete []last_shard_edge;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ShardedCSRGraph::import(EdgesListGraph &_el_graph)
{
    this->vertices_count = _el_graph.get_vertices_count();
    this->edges_count = _el_graph.get_edges_count();

    max_cached_vertices = 1*1024*1024/(sizeof(int));
    shards_number = (this->vertices_count - 1)/max_cached_vertices + 1;
    cout << "Shards number: " << shards_number << endl;

    resize(shards_number, this->vertices_count);

    import_direction(_el_graph, SCATTER);

    _el_graph.transpose();

    import_direction(_el_graph, GATHER);

    _el_graph.transpose();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


