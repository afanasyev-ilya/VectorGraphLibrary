#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::extract_connection_count(EdgesListGraph<_TVertexValue, _TEdgeWeight> &_el_graph,
                                                                             int *_work_buffer,
                                                                             int *_connections_array)
{
    int el_vertices_count = _el_graph.get_vertices_count();
    long long el_edges_count = _el_graph.get_edges_count();
    int *el_src_ids = _el_graph.get_src_ids();

    memset(_connections_array, 0, el_vertices_count*sizeof(int));
    memset(_work_buffer, 0, el_vertices_count*MAX_SX_AURORA_THREADS*sizeof(int));

    #pragma omp parallel
    {};

    double t1 = omp_get_wtime();
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        #pragma omp for
        for(long long vec_start = 0; vec_start < el_edges_count; vec_start += VECTOR_LENGTH)
        {
            #pragma _NEC vector
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                if((vec_start + i) < el_edges_count)
                {
                    int id = el_src_ids[vec_start + i];
                    _work_buffer[tid*el_vertices_count + id]++;
                }
            }
        }

        #pragma _NEC novector
        for(int core = 0; core < MAX_SX_AURORA_THREADS; core++)
        {
            #pragma _NEC ivdep
            #pragma omp for
            for(int i = 0; i < el_vertices_count; i++)
            {
                _connections_array[i] += _work_buffer[core *el_vertices_count + i];
            }
        }
    }
    double t2 = omp_get_wtime();
    cout << "extract connections time: " << t2 - t1 << " sec" << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::sort_vertices_by_degree(int *_connections_array,
                                                                            asl_int_t *_asl_indexes,
                                                                            int _el_vertices_count,
                                                                            int *_forward_conversion,
                                                                            int *_backward_conversion)
{
    double t1 = omp_get_wtime();
    // prepare indexes
    #pragma _NEC ivdep
    #pragma omp parallel for
    for(int i = 0; i < _el_vertices_count; i++)
    {
        _asl_indexes[i] = i;
    }

    ASL_CALL(asl_library_initialize());
    asl_sort_t hnd;
    ASL_CALL(asl_sort_create_i32(&hnd, ASL_SORTORDER_DESCENDING, ASL_SORTALGORITHM_AUTO));

    // do sorting
    ASL_CALL(asl_sort_execute_i32(hnd, _el_vertices_count, _connections_array, _asl_indexes, _connections_array, _asl_indexes));

    ASL_CALL(asl_sort_destroy(hnd));
    ASL_CALL(asl_library_finalize());

    #pragma _NEC ivdep
    #pragma omp parallel for
    for(int i = 0; i < _el_vertices_count; i++)
    {
        _forward_conversion[_asl_indexes[i]] = i;
    }

    #pragma _NEC ivdep
    #pragma omp parallel for
    for(int i = 0; i < _el_vertices_count; i++)
    {
        _backward_conversion[i] = _asl_indexes[i];
    }

    double t2 = omp_get_wtime();
    cout << "connections sorting (prepare reorder) time: " << t2 - t1 << " sec" << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::construct_CSR(EdgesListGraph<_TVertexValue, _TEdgeWeight> &_el_graph)
{
    double t1 = omp_get_wtime();

    int el_vertices_count = _el_graph.get_vertices_count();
    long long el_edges_count = _el_graph.get_edges_count();
    int *el_src_ids = _el_graph.get_src_ids();
    int *el_dst_ids = _el_graph.get_dst_ids();
    _TEdgeWeight *el_weights = _el_graph.get_weights();
    _TVertexValue *el_vertex_values = _el_graph.get_vertex_values();

    #pragma _NEC ivdep
    #pragma omp parallel
    for(long long cur_vertex = 0; cur_vertex < el_vertices_count; cur_vertex++)
    {
        this->outgoing_ptrs[cur_vertex] = -1;
        this->vertex_values[cur_vertex] = el_vertex_values[cur_vertex]; // TODO fix with reorder????
    }

    outgoing_ptrs[0] = 0;
    #pragma _NEC ivdep
    #pragma omp parallel
    for(long long cur_edge = 1; cur_edge < (el_edges_count - 1); cur_edge++)
    {
        int prev_id = el_src_ids[cur_edge - 1];
        int src_id = el_src_ids[cur_edge];
        if(src_id != prev_id)
            outgoing_ptrs[src_id] = cur_edge;
    }
    outgoing_ptrs[el_vertices_count] = el_edges_count;

    #pragma _NEC ivdep
    #pragma omp parallel // this can be done in parallel only because graph is sorted
    for(long long cur_vertex = el_vertices_count; cur_vertex >= 0; cur_vertex--)
    {
        if(this->outgoing_ptrs[cur_vertex] == -1) // if vertex has zero degree
        {
            this->outgoing_ptrs[cur_vertex] = el_edges_count; // since graph is sorted
        }
    }

    #pragma _NEC ivdep
    #pragma omp parallel
    for(long long cur_edge = 0; cur_edge < el_edges_count; cur_edge++)
    {
        this->outgoing_ids[cur_edge] = el_dst_ids[cur_edge];
        this->outgoing_weights[cur_edge] = el_weights[cur_edge];
    }

    double t2 = omp_get_wtime();
    cout << "CSR construction time: " << t2 - t1 << " sec" << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::import_and_preprocess(EdgesListGraph<_TVertexValue, _TEdgeWeight> &_el_graph)
{
    // get size of edges list graph
    int el_vertices_count = _el_graph.get_vertices_count();
    long long el_edges_count = _el_graph.get_edges_count();

    // reorder buffers
    int *loc_forward_conversion, *loc_backward_conversion;
    MemoryAPI::allocate_array(&loc_forward_conversion, el_vertices_count);
    MemoryAPI::allocate_array(&loc_backward_conversion, el_vertices_count);

    // allocate buffers
    int *connections_array;
    int *work_buffer;
    asl_int_t *asl_indexes;
    MemoryAPI::allocate_array(&connections_array, el_vertices_count);
    MemoryAPI::allocate_array(&asl_indexes, el_edges_count);
    MemoryAPI::allocate_array(&work_buffer, max(el_edges_count, (long long)el_vertices_count*MAX_SX_AURORA_THREADS));

    // obtain connections array from edges list graph
    extract_connection_count(_el_graph, work_buffer, connections_array);

    // get reorder data (sort by vertex degree)
    sort_vertices_by_degree(connections_array, asl_indexes, el_vertices_count, loc_forward_conversion,
                            loc_backward_conversion);
    MemoryAPI::free_array(connections_array);

    // reorder ids in edges list graph
    _el_graph.renumber_vertices(loc_forward_conversion, work_buffer);

    // sorting preprocessed edges list graph
    _el_graph.preprocess_into_csr_based(work_buffer, asl_indexes);
    MemoryAPI::allocate_array(&asl_indexes, el_edges_count);

    // resize constructed graph
    this->resize(el_vertices_count, el_edges_count);

    // construct CSR representation
    this->construct_CSR(_el_graph);

    // save conversion arrays into graph
    MemoryAPI::copy(forward_conversion, loc_forward_conversion, this->vertices_count);
    MemoryAPI::copy(backward_conversion, loc_backward_conversion, this->vertices_count);

    // return edges list graph to original state
    _el_graph.renumber_vertices(loc_backward_conversion, work_buffer);

    // free conversion arrays
    MemoryAPI::free_array(loc_forward_conversion);
    MemoryAPI::free_array(loc_backward_conversion);

    // free buffer
    MemoryAPI::free_array(work_buffer);

    #ifdef __USE_GPU__
    estimate_gpu_thresholds();
    #endif

    #ifdef __USE_NEC_SX_AURORA__
    estimate_nec_thresholds();
    last_vertices_ve.init_from_graph(this->outgoing_ptrs, this->outgoing_ids, this->outgoing_weights,
                                     vector_core_threshold_vertex, this->vertices_count);
    #endif
}

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
    
    //t1 = omp_get_wtime();
    for(long long int i = 0; i < tmp_edges_count; i++)
    {
        int src_id = old_src_ids[i];
        int dst_id = old_dst_ids[i];
        _TEdgeWeight weight = old_weights[i];

        if(_multiple_arcs_state == MULTIPLE_ARCS_REMOVED)
        {
            if(src_id == dst_id) // also remove self loops TODO fix
            {
                continue;
            }
        }
        
        if(_traversal_type == PUSH_TRAVERSAL)
            tmp_graph[src_id].push_back(TempEdgeData<_TEdgeWeight>(dst_id, weight));
        else if(_traversal_type == PULL_TRAVERSAL)
            tmp_graph[dst_id].push_back(TempEdgeData<_TEdgeWeight>(src_id, weight));
    }
    //t2 = omp_get_wtime();
    //cout << "creating intermediate representation time: " << t2 - t1 <<" sec" << endl;

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
    //t1 = omp_get_wtime();
    vector<pair<int, int> > pairs(tmp_vertices_count);
    for(int i = 0; i < tmp_vertices_count; i++)
        pairs[i] = make_pair(tmp_graph[i].size(), i);
    
    if(vertices_state == VERTICES_SORTED)
    {
        sort(pairs.begin(), pairs.end());
        reverse(pairs.begin(), pairs.end());
    }
    
    //t2 = omp_get_wtime();
    //cout << "sort time: " << t2 - t1 << " sec" << endl;
    
    // save old indexes array
    //t1 = omp_get_wtime();
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
    //t2 = omp_get_wtime();
    //cout << "index reordering time: " << t2 - t1 << " sec" << endl;
    
    // sort adjacent ids locally for each vertex
    long long no_loops_edges_count = 0;
    //t1 = omp_get_wtime();
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
    //t2 = omp_get_wtime();
    //cout << "edges sort time: " << t2 - t1 << " sec" << endl;
    
    // get new pointers
    //t1 = omp_get_wtime();
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

    //#ifdef __USE_NEC_SX_AURORA__
    estimate_nec_thresholds();
    last_vertices_ve.init_from_graph(this->outgoing_ptrs, this->outgoing_ids, this->outgoing_weights,
                                     vector_core_threshold_vertex, this->vertices_count);
    //#endif

    //t2 = omp_get_wtime();
    //cout << "final time: " << t2 - t1 << " sec" << endl << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
