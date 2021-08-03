/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSRGraph::construct_unsorted_csr(EdgesListGraph &_el_graph, bool _random_shuffle_required)
{
    if(_random_shuffle_required)
    {
        srand ( unsigned ( time(0) ) );

        vector<int> reorder_ids(_el_graph.get_vertices_count());
        for (int i = 0; i < _el_graph.get_vertices_count(); i++)
            reorder_ids[i] = i;
        random_shuffle (reorder_ids.begin(), reorder_ids.end() );

        for(long long i = 0; i < _el_graph.get_edges_count(); i++)
        {
            _el_graph.get_src_ids()[i] = reorder_ids[_el_graph.get_src_ids()[i]];
            _el_graph.get_dst_ids()[i] = reorder_ids[_el_graph.get_dst_ids()[i]];
        }
        cout << "CSRGraph::construct_unsorted_csr : random shuffle is done!" << endl;
    }

    int *work_buffer;
    vgl_sort_indexes *sort_indexes;
    MemoryAPI::allocate_array(&sort_indexes, this->edges_count);
    MemoryAPI::allocate_array(&work_buffer, max(this->edges_count, (long long)this->vertices_count*8));//TODO 8
    _el_graph.preprocess_into_csr_based(work_buffer, sort_indexes);

    #pragma omp parallel for
    for(int i = 0; i < this->vertices_count; i++)
        vertex_pointers[i] = -1;

    vertex_pointers[0] = 0;
    adjacent_ids[0] = _el_graph.get_dst_ids()[0];

    #pragma omp parallel for
    for(long long cur_edge = 1; cur_edge < this->edges_count; cur_edge++)
    {
        int src_id = _el_graph.get_src_ids()[cur_edge];
        int dst_id = _el_graph.get_dst_ids()[cur_edge];
        adjacent_ids[cur_edge] = /*rand() % this->vertices_count;*/ dst_id;

        int prev_id = _el_graph.get_src_ids()[cur_edge - 1];
        if(src_id != prev_id)
            vertex_pointers[src_id] = cur_edge;
    }
    vertex_pointers[this->vertices_count] = this->edges_count;

    #pragma omp parallel for
    for(long long cur_vertex = this->vertices_count; cur_vertex >= 0; cur_vertex--)
    {
        if(vertex_pointers[cur_vertex] == -1) // if vertex has zero degree
        {
            vertex_pointers[cur_vertex] = this->edges_count; // since graph is sorted
        }
    }

    MemoryAPI::free_array(work_buffer);
    MemoryAPI::free_array(sort_indexes);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSRGraph::import(EdgesListGraph &_el_graph, bool _random_shuffle_required)
{
    resize(_el_graph.get_vertices_count(), _el_graph.get_edges_count());
    construct_unsorted_csr(_el_graph, _random_shuffle_required);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
