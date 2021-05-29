#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ShardedCSRGraph::import_direction_2D_segmented(EdgesListGraph &_el_graph, TraversalDirection _import_direction)
{
    int *work_buffer;
    vgl_sort_indexes *edges_reorder_indexes;
    MemoryAPI::allocate_array(&work_buffer, _el_graph.get_edges_count());
    MemoryAPI::allocate_array(&edges_reorder_indexes, _el_graph.get_edges_count());

    _el_graph.transpose();
    _el_graph.preprocess_into_csr_based(work_buffer, edges_reorder_indexes);
    _el_graph.transpose(); // dst ids are sorted here

    MemoryAPI::free_array(work_buffer);

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
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_time_stats("Sharded split");
    #endif

    // import each shard from edges list
    tm.start();
    for(int shard_id = 0; shard_id < shards_number; shard_id++)
    {
        // import shard
        long long edges_in_shard = last_shard_edge[shard_id] - first_shard_edge[shard_id];
        long long first_shard_edge_val = first_shard_edge[shard_id];

        int *shard_src_ids_ptr = &(el_src_ids[first_shard_edge_val]);
        int *shard_dst_ids_ptr = &(el_dst_ids[first_shard_edge_val]);
        EdgesListGraph edges_list_shard;
        edges_list_shard.import(shard_src_ids_ptr, shard_dst_ids_ptr, this->vertices_count, edges_in_shard);

        if(_import_direction == SCATTER)
        {
            outgoing_shards[shard_id].import(edges_list_shard);
            outgoing_shards[shard_id].update_edge_reorder_indexes_using_superposition(&edges_reorder_indexes[first_shard_edge_val]);
        }
        else if(_import_direction == GATHER)
        {
            incoming_shards[shard_id].import(edges_list_shard);
            incoming_shards[shard_id].update_edge_reorder_indexes_using_superposition(&edges_reorder_indexes[first_shard_edge_val]);
        }
    }
    tm.end();
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_time_stats("Import shards");
    #endif

    delete []first_border;
    delete []last_border;
    delete []first_shard_edge;
    delete []last_shard_edge;

    MemoryAPI::free_array(edges_reorder_indexes);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ShardedCSRGraph::import_direction_random_segmenting(EdgesListGraph &_el_graph, TraversalDirection _import_direction)
{
    int *shards_for_vertex;
    MemoryAPI::allocate_array(&shards_for_vertex, _el_graph.get_vertices_count());
    MemoryAPI::set(shards_for_vertex, -1, _el_graph.get_vertices_count());

    vector<int> shard_guaranteed_vertex(shards_number);

    int shard_set_counter = 0;
    for(int i = 0; i < _el_graph.get_edges_count(); i++)
    {
        int src_id = _el_graph.get_src_ids()[i];
        int dst_id = _el_graph.get_dst_ids()[i];

        if(std::find(shard_guaranteed_vertex.begin(), shard_guaranteed_vertex.end(), src_id) != shard_guaranteed_vertex.end())
        {
            shard_guaranteed_vertex[shard_set_counter] = src_id;
            shard_set_counter++;
        }

        if(shard_set_counter >= shards_number)
        {
            break;
        }

        if(i == (_el_graph.get_edges_count() - 1))
        {
            throw "Error: not enough graph vertices for specified shard count in ShardedCSRGraph::import_direction_random_segmenting";
        }
    }

    for(int i = 0; i < _el_graph.get_vertices_count(); i++)
    {
        shards_for_vertex[i] = rand() % shards_number;
    }

    #ifdef __USE_MPI__
    vgl_library_data.bcast(shards_for_vertex, _el_graph.get_vertices_count(), 0);
    #endif

    for(int i = 0; i < shard_guaranteed_vertex.size(); i++)
    {
        shards_for_vertex[shard_guaranteed_vertex[i]] = i;
    }

    for(int shard_id = 0; shard_id < shards_number; shard_id++)
    {
        vector<int> src_ids_vec;
        vector<int> dst_ids_vec;

        size_t edges_in_shard = 0;

        for(size_t i = 0; i < _el_graph.get_edges_count(); i++)
        {
            int src_id = _el_graph.get_src_ids()[i];
            int dst_id = _el_graph.get_dst_ids()[i];
            if(shards_for_vertex[src_id] == shard_id)
            {
                src_ids_vec.push_back(src_id);
                dst_ids_vec.push_back(dst_id);
                edges_in_shard++;
            }
        }

        EdgesListGraph edges_list_shard;
        edges_list_shard.import(&src_ids_vec[0], &dst_ids_vec[0], _el_graph.get_vertices_count(), edges_in_shard);

        if(_import_direction == SCATTER)
        {
            outgoing_shards[shard_id].import(edges_list_shard);
            //outgoing_shards[shard_id].update_edge_reorder_indexes_using_superposition(&edges_reorder_indexes[first_shard_edge_val]);
            cout << "outgoing shard: " << shard_id << " v=" << outgoing_shards[shard_id].get_vertices_count() << " e="
                 << outgoing_shards[shard_id].get_edges_count() << endl;
        }
        else if(_import_direction == GATHER)
        {
            incoming_shards[shard_id].import(edges_list_shard);
            cout << "incoming shard: " << shard_id << " v=" << incoming_shards[shard_id].get_vertices_count() << " e="
                 << incoming_shards[shard_id].get_edges_count() << endl;
        }
    }

    /*for(int shard_id = 0; shard_id < shards_number; shard_id++)
    {
        vector<int> src_ids_vec;
        vector<int> dst_ids_vec;

        size_t edges_in_shard = 0;

        for(int src_id = 0; src_id < whole_graph.get_vertices_count(); src_id++)
        {
            if(shards_for_vertex[src_id] == shard_id)
            {
                int connections = whole_graph.get_connections_count(src_id);
                for(int i = 0; i < connections; i++)
                {
                    int dst_id = whole_graph.get_edge_dst(src_id, i);
                    src_ids_vec.push_back(src_id);
                    dst_ids_vec.push_back(dst_id);
                }
                edges_in_shard += connections;
            }
        }

        EdgesListGraph edges_list_shard;
        edges_list_shard.import(&src_ids_vec[0], &dst_ids_vec[0], _el_graph.get_vertices_count(), edges_in_shard);
        outgoing_shards[shard_id].import(edges_list_shard);

        cout << "shard: " << shard_id << " v=" << outgoing_shards[shard_id].get_vertices_count() << " e="
        << outgoing_shards[shard_id].get_edges_count() << endl;
    }*/
    cout << "whole old: " << " v=" << _el_graph.get_vertices_count() << " e=" << _el_graph.get_edges_count() << endl;

    MemoryAPI::free_array(shards_for_vertex);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ShardedCSRGraph::import(EdgesListGraph &_el_graph, int _force_shards_number)
{
    this->vertices_count = _el_graph.get_vertices_count();
    this->edges_count = _el_graph.get_edges_count();

    if(_force_shards_number == 0)
    {
        max_cached_vertices = 8*1024*1024/(sizeof(int));
        shards_number = (this->vertices_count - 1)/max_cached_vertices + 1;
        cout << "Shards number: " << shards_number << endl;

        resize(shards_number, this->vertices_count);

        if(can_use_scatter())
        {
            import_direction_2D_segmented(_el_graph, SCATTER);
        }

        if(can_use_gather())
        {
            _el_graph.transpose();
            import_direction_2D_segmented(_el_graph, GATHER);
            _el_graph.transpose();
        }
    }
    else
    {
        shards_number = _force_shards_number;
        cout << "Shards number: " << shards_number << endl;

        resize(shards_number, this->vertices_count);

        if(can_use_scatter())
        {
            import_direction_random_segmenting(_el_graph, SCATTER);
        }

        if(can_use_gather())
        {
            _el_graph.transpose();
            import_direction_random_segmenting(_el_graph, GATHER);
            _el_graph.transpose();
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


