/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSRGraph::construct_unsorted_csr(EdgesContainer &_edges_container)
{
    int *work_buffer;
    vgl_sort_indexes *sort_indexes;
    MemoryAPI::allocate_array(&sort_indexes, this->edges_count);
    MemoryAPI::allocate_array(&work_buffer, max(this->edges_count, (long long)this->vertices_count*8));//TODO 8
    _edges_container.preprocess_into_csr_based(work_buffer, sort_indexes);

    this->copy_edges_indexes(sort_indexes);

    #pragma omp parallel for
    for(int i = 0; i < this->vertices_count; i++)
        vertex_pointers[i] = -1;

    vertex_pointers[0] = 0;
    adjacent_ids[0] = _edges_container.get_dst_ids()[0];

    // if first edge is (first_edge_src, something) we need to put additional zeroes to first vertex pointers positions
    int first_edge_src = _edges_container.get_src_ids()[0];
    if(first_edge_src != 0)
        for(int i = 0; i <= first_edge_src; i++)
            vertex_pointers[i] = 0;

    #pragma omp parallel for
    for(long long cur_edge = 1; cur_edge < this->edges_count; cur_edge++)
    {
        int src_id = _edges_container.get_src_ids()[cur_edge];
        int dst_id = _edges_container.get_dst_ids()[cur_edge];
        adjacent_ids[cur_edge] = dst_id;

        int prev_id = _edges_container.get_src_ids()[cur_edge - 1];
        if(src_id != prev_id)
            vertex_pointers[src_id] = cur_edge;
    }
    vertex_pointers[this->vertices_count] = this->edges_count;

    // must be sequential!
    for(long long cur_vertex = (vertices_count - 1); cur_vertex != 0; cur_vertex--)
    {
        if(vertex_pointers[cur_vertex] == -1)
            vertex_pointers[cur_vertex] = vertex_pointers[cur_vertex + 1];
    }

    MemoryAPI::free_array(work_buffer);
    MemoryAPI::free_array(sort_indexes);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSRGraph::copy_edges_indexes(vgl_sort_indexes *_sort_indexes)
{
    #pragma omp parallel for
    for(long long i = 0; i < this->edges_count; i++)
    {
        edges_reorder_indexes[i] = _sort_indexes[i];
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSRGraph::import(EdgesContainer &_edges_container)
{
    resize(_edges_container.get_vertices_count(), _edges_container.get_edges_count());
    construct_unsorted_csr(_edges_container);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
