#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void UndirectedVectCSRGraph::extract_connection_count(EdgesListGraph &_el_graph,
                                                  int *_work_buffer,
                                                  int *_connections_array,
                                                  int _threads_num)
{
    Timer tm;
    tm.start();

    int el_vertices_count = _el_graph.get_vertices_count();
    long long el_edges_count = _el_graph.get_edges_count();
    int *el_src_ids = _el_graph.get_src_ids();

    memset(_connections_array, 0, el_vertices_count*sizeof(int));
    memset(_work_buffer, 0, el_vertices_count*_threads_num*sizeof(int));

    #pragma omp parallel
    {};

    #pragma omp parallel num_threads(_threads_num)
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
        for(int core = 0; core < _threads_num; core++)
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

void UndirectedVectCSRGraph::sort_vertices_by_degree(int *_connections_array,
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

void UndirectedVectCSRGraph::construct_CSR(EdgesListGraph &_el_graph)
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

void UndirectedVectCSRGraph::copy_edges_indexes(vgl_sort_indexes *_sort_indexes)
{
    #pragma _NEC ivdep
    #pragma omp parallel for
    for(long long i = 0; i < this->edges_count; i++)
    {
        edges_reorder_indexes[i] = _sort_indexes[i];
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void UndirectedVectCSRGraph::remove_loops_and_multiple_arcs()
{
    int *new_connections_count, *new_adjacent_ids;
    long long *new_vertex_pointers;
    MemoryAPI::allocate_array(&new_connections_count, this->vertices_count);
    MemoryAPI::allocate_array(&new_vertex_pointers, this->vertices_count + 1);

    // calculate duplicates count and new connections cound
    #pragma omp parallel for
    for(int src_id = 0; src_id < this->vertices_count; src_id++)
    {
        long long start = this->vertex_pointers[src_id];
        long long end = this->vertex_pointers[src_id + 1];
        int connections_count = end - start;
        int duplicates_count = 0;
        for(long long cur_edge = start + 1; cur_edge < end; cur_edge++)
        {
            if((this->adjacent_ids[cur_edge] == this->adjacent_ids[cur_edge - 1]))
                duplicates_count++;
        }

        new_connections_count[src_id] = connections_count - duplicates_count;
    }

    // calculate new edges count
    long long new_edges_count = 0;
    #pragma omp parallel for reduction(+: new_edges_count)
    for(int src_id = 0; src_id < this->vertices_count; src_id++)
    {
        new_edges_count += new_connections_count[src_id];
    }

    cout << "UndirectedVectCSRGraph::remove_loops_and_multiple_arcs reduced edges from " << this->edges_count << " to " << new_edges_count << endl;
    MemoryAPI::allocate_array(&new_adjacent_ids, new_edges_count);

    // obtain new vertex pointers
    new_vertex_pointers[0] = 0;
    for(int src_id = 1; src_id < this->vertices_count + 1; src_id++) // TODO in parallel
    {
        new_vertex_pointers[src_id] = new_vertex_pointers[src_id - 1] + new_connections_count[src_id - 1];
    }

    // free tmp connections array
    MemoryAPI::free_array(new_connections_count);

    // copy edges
    #pragma omp parallel for
    for(int src_id = 0; src_id < this->vertices_count; src_id++)
    {
        long long start = this->vertex_pointers[src_id];
        long long end = this->vertex_pointers[src_id + 1];

        long long new_dst = new_vertex_pointers[src_id];

        for(long long old_pos = start; old_pos < end; old_pos++)
        {
            if(old_pos == start)
            {
                new_adjacent_ids[new_dst] = this->adjacent_ids[old_pos];
                new_dst++;
            }
            else
            {
                if((this->adjacent_ids[old_pos] == this->adjacent_ids[old_pos - 1]))
                {
                    continue;
                }
                else
                {
                    new_adjacent_ids[new_dst] = this->adjacent_ids[old_pos];
                    new_dst++;
                }
            }
        }
    }

    // free old data
    MemoryAPI::free_array(new_vertex_pointers);
    MemoryAPI::free_array(new_adjacent_ids);

    // copy new data into graph
    this->edges_count = new_edges_count;
    this->vertex_pointers = new_vertex_pointers;
    this->adjacent_ids = new_adjacent_ids;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void UndirectedVectCSRGraph::import(EdgesListGraph &_el_graph)
{
    // get size of edges list graph
    int el_vertices_count = _el_graph.get_vertices_count();
    long long el_edges_count = _el_graph.get_edges_count();

    // reorder buffers
    int *loc_forward_conversion, *loc_backward_conversion;
    MemoryAPI::allocate_array(&loc_forward_conversion, el_vertices_count);
    MemoryAPI::allocate_array(&loc_backward_conversion, el_vertices_count);

    int max_threads_in_extract = 1;
    #ifdef __USE_MULTICORE__
    // if arch have many cores, we don't use many threads in order to prevent huge memory consumption.
    // instead, we use edge factor threads
    int edge_factor = el_edges_count/el_vertices_count;
    max_threads_in_extract = edge_factor;
    #endif

    #ifdef __USE_NEC_SX_AURORA__
    max_threads_in_extract = omp_get_max_threads();
    #endif

    // allocate buffers
    int *connections_array;
    int *work_buffer;
    vgl_sort_indexes *sort_indexes;
    MemoryAPI::allocate_array(&connections_array, el_vertices_count);
    MemoryAPI::allocate_array(&sort_indexes, el_edges_count);
    // this buffer should have enough elements for extracted connections and edges list items sort
    MemoryAPI::allocate_array(&work_buffer, max(el_edges_count, (long long)el_vertices_count*max_threads_in_extract));

    // obtain connections array from edges list graph
    extract_connection_count(_el_graph, work_buffer, connections_array, max_threads_in_extract);

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

    // sort edges
    this->sort_adjacent_edges();

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

    #if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
    estimate_nec_thresholds();
    last_vertices_ve.init_from_graph(this->vertex_pointers, this->adjacent_ids,
                                     vector_core_threshold_vertex, this->vertices_count);
    #ifdef __USE_MPI__
    estimate_mpi_thresholds();
    #endif
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void UndirectedVectCSRGraph::sort_adjacent_edges()
{
    for(long long cur_vertex = 0; cur_vertex < this->vertices_count; cur_vertex++)
    {
        int connections_count = this->vertex_pointers[cur_vertex + 1] - this->vertex_pointers[cur_vertex];
        long long start = this->vertex_pointers[cur_vertex];
        if(connections_count >= 2)
        {
            Sorter::sort(&(this->adjacent_ids[start]), NULL, connections_count, SORT_ASCENDING);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
