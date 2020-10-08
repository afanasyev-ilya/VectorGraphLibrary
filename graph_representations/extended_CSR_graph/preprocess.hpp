#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ExtendedCSRGraph::extract_connection_count(EdgesListGraph &_el_graph,
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

void ExtendedCSRGraph::sort_vertices_by_degree(int *_connections_array,
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

void ExtendedCSRGraph::construct_CSR(EdgesListGraph &_el_graph)
{
    double t1 = omp_get_wtime();

    int el_vertices_count = _el_graph.get_vertices_count();
    long long el_edges_count = _el_graph.get_edges_count();
    int *el_src_ids = _el_graph.get_src_ids();
    int *el_dst_ids = _el_graph.get_dst_ids();

    #pragma _NEC ivdep
    #pragma omp parallel
    for(long long cur_vertex = 0; cur_vertex < el_vertices_count; cur_vertex++)
    {
        this->vertex_pointers[cur_vertex] = -1;
    }

    vertex_pointers[0] = 0;
    #pragma _NEC ivdep
    #pragma omp parallel
    for(long long cur_edge = 1; cur_edge < el_edges_count; cur_edge++)
    {
        int prev_id = el_src_ids[cur_edge - 1];
        int src_id = el_src_ids[cur_edge];
        if(src_id != prev_id)
            vertex_pointers[src_id] = cur_edge;
    }
    vertex_pointers[el_vertices_count] = el_edges_count;

    #pragma _NEC ivdep
    #pragma omp parallel // this can be done in parallel only because graph is sorted
    for(long long cur_vertex = el_vertices_count; cur_vertex >= 0; cur_vertex--)
    {
        if(this->vertex_pointers[cur_vertex] == -1) // if vertex has zero degree
        {
            this->vertex_pointers[cur_vertex] = el_edges_count; // since graph is sorted
        }
    }

    #pragma _NEC ivdep
    #pragma omp parallel
    for(long long cur_edge = 0; cur_edge < el_edges_count; cur_edge++)
    {
        this->adjacent_ids[cur_edge] = el_dst_ids[cur_edge];
    }

    double t2 = omp_get_wtime();
    cout << "CSR construction time: " << t2 - t1 << " sec" << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ExtendedCSRGraph::import_and_preprocess(EdgesListGraph &_el_graph)
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
    last_vertices_ve.init_from_graph(this->vertex_pointers, this->adjacent_ids,
                                     vector_core_threshold_vertex, this->vertices_count);
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
