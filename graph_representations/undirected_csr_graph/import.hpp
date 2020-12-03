#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void UndirectedCSRGraph::extract_connection_count(EdgesListGraph &_el_graph,
                                                  int *_work_buffer,
                                                  int *_connections_array)
{
    Timer tm;
    tm.start();

    int el_vertices_count = _el_graph.get_vertices_count();
    long long el_edges_count = _el_graph.get_edges_count();
    int *el_src_ids = _el_graph.get_src_ids();

    memset(_connections_array, 0, el_vertices_count*sizeof(int));
    memset(_work_buffer, 0, el_vertices_count*MAX_SX_AURORA_THREADS*sizeof(int));

    #pragma omp parallel
    {};

    #pragma omp parallel num_threads(MAX_SX_AURORA_THREADS)
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

    tm.end();
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_time_stats("Extract connections count from EdgesListGraph");
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void UndirectedCSRGraph::sort_vertices_by_degree(int *_connections_array,
                                                 vgl_sort_indexes *_sort_indexes,
                                                 int _el_vertices_count,
                                                 int *_forward_conversion,
                                                 int *_backward_conversion)
{
    Timer tm;
    tm.start();

    // prepare indexes
    #pragma _NEC ivdep
    #pragma omp parallel for
    for(int i = 0; i < _el_vertices_count; i++)
    {
        _sort_indexes[i] = i;
    }

    // sorting
    Sorter::sort(_connections_array, _sort_indexes, _el_vertices_count, SORT_DESCENDING);

    #pragma _NEC ivdep
    #pragma omp parallel for
    for(int i = 0; i < _el_vertices_count; i++)
    {
        _forward_conversion[_sort_indexes[i]] = i;
    }

    #pragma _NEC ivdep
    #pragma omp parallel for
    for(int i = 0; i < _el_vertices_count; i++)
    {
        _backward_conversion[i] = _sort_indexes[i];
    }

    tm.end();
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_time_stats("connections sorting (prepare reorder)");
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void UndirectedCSRGraph::construct_CSR(EdgesListGraph &_el_graph)
{
    Timer tm;
    tm.start();

    int el_vertices_count = _el_graph.get_vertices_count();
    long long el_edges_count = _el_graph.get_edges_count();
    int *el_src_ids = _el_graph.get_src_ids();
    int *el_dst_ids = _el_graph.get_dst_ids();

    #pragma _NEC ivdep
    #pragma omp parallel for
    for(long long cur_vertex = 0; cur_vertex < el_vertices_count; cur_vertex++)
    {
        this->vertex_pointers[cur_vertex] = -1;
    }

    vertex_pointers[0] = 0;
    #pragma _NEC ivdep
    #pragma omp parallel for
    for(long long cur_edge = 1; cur_edge < el_edges_count; cur_edge++)
    {
        int prev_id = el_src_ids[cur_edge - 1];
        int src_id = el_src_ids[cur_edge];
        if(src_id != prev_id)
            vertex_pointers[src_id] = cur_edge;
    }
    vertex_pointers[el_vertices_count] = el_edges_count;

    #pragma _NEC ivdep
    #pragma omp parallel for// this can be done in parallel only because graph is sorted
    for(long long cur_vertex = el_vertices_count; cur_vertex >= 0; cur_vertex--)
    {
        if(this->vertex_pointers[cur_vertex] == -1) // if vertex has zero degree
        {
            this->vertex_pointers[cur_vertex] = el_edges_count; // since graph is sorted
        }
    }

    #pragma _NEC ivdep
    #pragma omp parallel for
    for(long long cur_edge = 0; cur_edge < el_edges_count; cur_edge++)
    {
        this->adjacent_ids[cur_edge] = el_dst_ids[cur_edge];
    }

    tm.end();
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_time_stats("CSR construction");
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void UndirectedCSRGraph::copy_edges_indexes(vgl_sort_indexes *_sort_indexes)
{
    #pragma _NEC ivdep
    #pragma omp parallel for
    for(long long i = 0; i < this->edges_count; i++)
    {
        edges_reorder_indexes[i] = _sort_indexes[i];
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void UndirectedCSRGraph::import(EdgesListGraph &_el_graph)
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
    vgl_sort_indexes *sort_indexes;
    MemoryAPI::allocate_array(&connections_array, el_vertices_count);
    MemoryAPI::allocate_array(&sort_indexes, el_edges_count);
    MemoryAPI::allocate_array(&work_buffer, max(el_edges_count, (long long)el_vertices_count*MAX_SX_AURORA_THREADS));

    // obtain connections array from edges list graph
    extract_connection_count(_el_graph, work_buffer, connections_array);

    // get reorder data (sort by vertex degree)
    sort_vertices_by_degree(connections_array, sort_indexes, el_vertices_count, loc_forward_conversion,
                            loc_backward_conversion);
    MemoryAPI::free_array(connections_array);

    // reorder ids in edges list graph
    _el_graph.renumber_vertices(loc_forward_conversion, work_buffer);

    // sorting preprocessed edges list graph
    _el_graph.preprocess_into_csr_based(work_buffer, sort_indexes);

    // resize constructed graph
    this->resize(el_vertices_count, el_edges_count);

    // save reordering information and free ASL array
    this->copy_edges_indexes(sort_indexes);

    MemoryAPI::free_array(sort_indexes);

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

    #ifdef __USE_NEC_SX_AURORA__
    estimate_nec_thresholds();
    last_vertices_ve.init_from_graph(this->vertex_pointers, this->adjacent_ids,
                                     vector_core_threshold_vertex, this->vertices_count);
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////