#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::import_graph(EdgesListGraph<_TVertexValue, _TEdgeWeight> &_old_graph,
                                                                 VerticesState _vertices_state,
                                                                 EdgesState _edges_state,
                                                                 int _supported_vector_length,
                                                                 TraversalDirection _traversal_type,
                                                                 MultipleArcsState _multiple_arcs_state)
{
    double t1, t2;
    
    // set optimisation parameters
    this->vertices_state          = _vertices_state;
    this->edges_state             = _edges_state;
    this->supported_vector_length = _supported_vector_length;
    
    // create tmp graph
    int tmp_vertices_count = _old_graph.get_vertices_count();
    long long tmp_edges_count = _old_graph.get_edges_count();
    
    vector<vector<TempEdgeData<_TEdgeWeight> > >tmp_graph(tmp_vertices_count);
    
    _TVertexValue *old_vertex_values = _old_graph.get_vertex_values();
    int *old_src_ids = _old_graph.get_src_ids();
    int *old_dst_ids = _old_graph.get_dst_ids();
    _TEdgeWeight *old_weights = _old_graph.get_weights();
    
    t1 = omp_get_wtime();
    for(long long int i = 0; i < tmp_edges_count; i++)
    {
        int src_id = old_src_ids[i];
        int dst_id = old_dst_ids[i];
        _TEdgeWeight weight = old_weights[i];
        
        if(_traversal_type == PUSH_TRAVERSAL)
            tmp_graph[src_id].push_back(TempEdgeData<_TEdgeWeight>(dst_id, weight));
        else if(_traversal_type == PULL_TRAVERSAL)
            tmp_graph[dst_id].push_back(TempEdgeData<_TEdgeWeight>(src_id, weight));
    }
    t2 = omp_get_wtime();
    cout << "creating intermediate representation time: " << t2 - t1 <<" sec" << endl;

    // remove multiple arcs here, since sorting is required to be maintained
    if(_multiple_arcs_state == MULTIPLE_ARCS_REMOVED)
    {
        // remove multiple arcs
        #pragma omp parallel for
        for(int cur_vertex = 0; cur_vertex < tmp_vertices_count; cur_vertex++)
        {
            int src_id = cur_vertex;
            std::sort(tmp_graph[src_id].begin(), tmp_graph[src_id].end(), edge_less < _TEdgeWeight > );
            tmp_graph[src_id].erase(unique(tmp_graph[src_id].begin(), tmp_graph[src_id].end(),
                                    edge_equal < _TEdgeWeight > ), tmp_graph[src_id].end());
        }

        // compute new edges count (since some edges could be removed)
        tmp_edges_count = 0;
        for(int cur_vertex = 0; cur_vertex < tmp_vertices_count; cur_vertex++)
        {
            tmp_edges_count += tmp_graph[cur_vertex].size();
        }
    }
    
    // sort all vertices now
    t1 = omp_get_wtime();
    vector<pair<int, int> > pairs(tmp_vertices_count);
    for(int i = 0; i < tmp_vertices_count; i++)
        pairs[i] = make_pair(tmp_graph[i].size(), i);
    
    if(vertices_state == VERTICES_SORTED)
    {
        sort(pairs.begin(), pairs.end());
        reverse(pairs.begin(), pairs.end());
    }
    
    t2 = omp_get_wtime();
    cout << "sort time: " << t2 - t1 << " sec" << endl;
    
    // save old indexes array
    t1 = omp_get_wtime();
    int *old_indexes;
    MemoryAPI::allocate_array(&old_indexes, tmp_vertices_count);
    for(int i = 0; i < tmp_vertices_count; i++)
    {
        old_indexes[i] = pairs[i].second;
    }
    
    // need to reoerder all data arrays in 2 steps
    vector<vector<TempEdgeData<_TEdgeWeight> > > new_tmp_graph(tmp_vertices_count);
    #pragma omp parallel for
    for(int i = 0; i < tmp_vertices_count; i++)
    {
        new_tmp_graph[i] = tmp_graph[old_indexes[i]];
    }

    #pragma omp parallel for
    for(int i = 0; i < tmp_vertices_count; i++)
    {
        tmp_graph[i] = new_tmp_graph[i];
    }
    
    // get correct reordered array
    int *tmp_reordered_vertex_ids;
    MemoryAPI::allocate_array(&tmp_reordered_vertex_ids, tmp_vertices_count);
    for(int i = 0; i < tmp_vertices_count; i++)
    {
        tmp_reordered_vertex_ids[old_indexes[i]] = i;
    }

    MemoryAPI::free_array(old_indexes);
    t2 = omp_get_wtime();
    cout << "index reordering time: " << t2 - t1 << " sec" << endl;
    
    // sort adjacent ids locally for each vertex
    long long no_loops_edges_count = 0;
    t1 = omp_get_wtime();
    #pragma omp parallel for
    for(int cur_vertex = 0; cur_vertex < tmp_vertices_count; cur_vertex++)
    {
        int src_id = cur_vertex;
        for(int i = 0; i < tmp_graph[src_id].size(); i++)
        {
            tmp_graph[src_id][i].dst_id = tmp_reordered_vertex_ids[tmp_graph[src_id][i].dst_id];
        }
        if(edges_state == EDGES_SORTED)
        {
            std::sort(tmp_graph[src_id].begin(), tmp_graph[src_id].end(), edge_less<_TEdgeWeight>);
        }
        else if(edges_state == EDGES_RANDOM_SHUFFLED)
        {
            std::random_shuffle(tmp_graph[src_id].begin(), tmp_graph[src_id].end());
        }
    }
    t2 = omp_get_wtime();
    cout << "edges sort time: " << t2 - t1 << " sec" << endl;
    
    // get new pointers
    t1 = omp_get_wtime();
    this->resize(tmp_vertices_count, tmp_edges_count);
    
    // save optimised graph
    long long current_edge = 0;
    this->outgoing_ptrs[0] = current_edge;
    for(int cur_vertex = 0; cur_vertex < this->vertices_count; cur_vertex++)
    {
        int src_id = cur_vertex;
        this->vertex_values[cur_vertex] = old_vertex_values[cur_vertex];
        this->reordered_vertex_ids[cur_vertex] = tmp_reordered_vertex_ids[cur_vertex];
        
        for(int i = 0; i < tmp_graph[src_id].size(); i++)
        {
            this->outgoing_ids[current_edge] = tmp_graph[src_id][i].dst_id;
            this->outgoing_weights[current_edge] = tmp_graph[src_id][i].weight;
            current_edge++;
        }
        this->outgoing_ptrs[cur_vertex + 1] = current_edge;
    }

    MemoryAPI::free_array(tmp_reordered_vertex_ids);
    
    calculate_incoming_degrees();
    
    #ifdef __USE_GPU__
    estimate_gpu_thresholds();
    #endif

    #ifdef __USE_NEC_SX_AURORA__
    estimate_nec_thresholds();
    last_vertices_ve.init_from_graph(this->outgoing_ptrs, this->outgoing_ids, this->outgoing_weights,
                                     nec_vector_core_threshold_vertex, this->vertices_count);
    #endif

    t2 = omp_get_wtime();
    cout << "final time: " << t2 - t1 << " sec" << endl << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::calculate_incoming_degrees()
{
    int vertices_count = this->vertices_count;
    for(int i = 0; i < vertices_count; i++)
    {
        incoming_degrees[i] = 0;
    }
    
    for(int src_id = 0; src_id < vertices_count; src_id++)
    {
        long long edge_start = outgoing_ptrs[src_id];
        int connections_count = outgoing_ptrs[src_id + 1] - outgoing_ptrs[src_id];
        for(int edge_pos = 0; edge_pos < connections_count; edge_pos++)
        {
            int dst_id = outgoing_ids[edge_start + edge_pos];
            incoming_degrees[dst_id]++;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
