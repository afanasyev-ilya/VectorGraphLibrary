/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSRGraph::construct_unsorted_csr(EdgesContainer &_edges_container)
{
    int *work_buffer;
    vgl_sort_indexes *sort_indexes;
    MemoryAPI::allocate_array(&sort_indexes, this->edges_count);
    MemoryAPI::allocate_array(&work_buffer, max(this->edges_count, (long long)this->vertices_count*8));//TODO 8
    _edges_container.preprocess_into_csr_based(work_buffer, sort_indexes);

    #pragma omp parallel for
    for(int i = 0; i < this->vertices_count; i++)
        vertex_pointers[i] = -1;

    vertex_pointers[0] = 0;
    adjacent_ids[0] = _edges_container.get_dst_ids()[0];

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

    long long cnt = 0;
    for(long long cur_vertex = 0; cur_vertex < (vertices_count + 1); cur_vertex++)
    {
        if(vertex_pointers[cur_vertex] != -1)
            cnt = vertex_pointers[cur_vertex];
        if(vertex_pointers[cur_vertex] == -1)
        {
            vertex_pointers[cur_vertex] = cnt;
        }
    }

    MemoryAPI::free_array(work_buffer);
    MemoryAPI::free_array(sort_indexes);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSRGraph::import(EdgesContainer &_edges_container)
{
    resize(_edges_container.get_vertices_count(), _edges_container.get_edges_count());
    construct_unsorted_csr(_edges_container);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
