/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSRGraph::import(EdgesListGraph &_el_graph)
{
    resize(_el_graph.get_vertices_count(), _el_graph.get_edges_count());

    int *work_buffer;
    vgl_sort_indexes *sort_indexes;
    MemoryAPI::allocate_array(&sort_indexes, this->edges_count);
    MemoryAPI::allocate_array(&work_buffer, max(this->edges_count, (long long)this->vertices_count*8));//TODO 8
    _el_graph.preprocess_into_csr_based(work_buffer, sort_indexes);

    for(int i = 0; i < this->vertices_count; i++)
        vertex_pointers[i] = -1;

    vertex_pointers[0] = 0;
    adjacent_ids[0] = _el_graph.get_dst_ids()[0];
    for(long long cur_edge = 1; cur_edge < this->edges_count; cur_edge++)
    {
        int src_id = _el_graph.get_src_ids()[cur_edge];
        int dst_id = _el_graph.get_dst_ids()[cur_edge];
        adjacent_ids[cur_edge] = rand() % this->vertices_count;//dst_id;

        int prev_id = _el_graph.get_src_ids()[cur_edge - 1];
        if(src_id != prev_id)
            vertex_pointers[src_id] = cur_edge;
    }
    vertex_pointers[this->vertices_count] = this->edges_count;

    for(long long cur_vertex = this->vertices_count; cur_vertex >= 0; cur_vertex--)
    {
        if(vertex_pointers[cur_vertex] == -1) // if vertex has zero degree
        {
            vertex_pointers[cur_vertex] = this->edges_count; // since graph is sorted
        }
    }

    MemoryAPI::free_array(work_buffer);
    MemoryAPI::free_array(sort_indexes);

    for(int i = 0; i < 10; i++)
        cout << vertex_pointers[i + 1] - vertex_pointers[i] << endl;

    int medium_degree_num = 0;
    int small_128_degree_num = 0;
    long long adj_num = 0;
    long long adj_small_128_num = 0;
    for(int src_id = 0; src_id < this->vertices_count; src_id++)
    {
        int con_count = vertex_pointers[src_id + 1] - vertex_pointers[src_id];
        if((con_count >= 256))
        {
            adj_num += con_count;
            medium_degree_num++;
        }

        if((con_count >= 32) && (con_count < 256))
        {
            adj_small_128_num += con_count;
            small_128_degree_num++;
        }
    }

    cout << "medium_degree_num: " << medium_degree_num << endl;
    cout << "small_32_degree_num: " << small_128_degree_num << endl;

    int *medium_degree_ids = new int[medium_degree_num];
    int *small_128_degree_ids = new int[small_128_degree_num];
    int pos = 0;
    int pos_small_128 = 0;
    for(int src_id = 0; src_id < this->vertices_count; src_id++)
    {
        int con_count = vertex_pointers[src_id + 1] - vertex_pointers[src_id];
        if((con_count >= 256))
        {
            medium_degree_ids[pos] = src_id;
            pos++;
        }
        if((con_count >= 32) && (con_count < 256))
        {
            small_128_degree_ids[pos_small_128] = src_id;
            pos_small_128++;
        }
    }

    int *data = new int[this->vertices_count];
    int *result = new int[this->edges_count];

    #pragma omp parallel
    {};

    double t1 = omp_get_wtime();
    #pragma omp parallel for schedule(static, 8)
    for(int i = 0; i < medium_degree_num; i++)
    {
        int src_id = medium_degree_ids[i];
        long long first = vertex_pointers[src_id];
        long long last = vertex_pointers[src_id + 1];

        #pragma _NEC cncall
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vob
        #pragma _NEC vector
        #pragma _NEC gather_reorder
        for(long long edge_pos = first; edge_pos < last; edge_pos++)
        {
            int dst_id = adjacent_ids[edge_pos];
            result[edge_pos] = data[dst_id];
        }
    }
    double t2 = omp_get_wtime();
    cout << "BW large: " << adj_num * sizeof(int)*3.0 / ((t2 - t1) * 1e9) << " GB/s" << endl;

    t1 = omp_get_wtime();
    #pragma omp parallel for
    for(int i = 0; i < small_128_degree_num; i++)
    {
        int src_id = small_128_degree_ids[i];
        long long first = vertex_pointers[src_id];
        long long last = vertex_pointers[src_id + 1];

        #pragma _NEC cncall
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vob
        #pragma _NEC vector
        #pragma _NEC gather_reorder
        for(long long edge_pos = first; edge_pos < last; edge_pos++)
        {
            int dst_id = adjacent_ids[edge_pos];
            result[edge_pos] = data[dst_id];
        }
    }
    t2 = omp_get_wtime();
    cout << "BW small 32: " << adj_small_128_num * sizeof(int)*3.0 / ((t2 - t1) * 1e9) << " GB/s" << endl;

    t1 = omp_get_wtime();
    #pragma omp parallel for
    for(int i = 0; i < small_128_degree_num; i++)
    {
        int src_id = small_128_degree_ids[i];
        long long first = vertex_pointers[src_id];
        long long last = vertex_pointers[src_id + 1];

        #pragma _NEC cncall
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vob
        #pragma _NEC vector
        #pragma _NEC gather_reorder
        for(int i = 0; i < 256; i++)
        {
            long long edge_pos = first + i;
            if(edge_pos < last)
            {
                int dst_id = adjacent_ids[edge_pos];
                result[edge_pos] = data[dst_id];
            }
        }
    }
    t2 = omp_get_wtime();
    cout << "BW small fixed VL 32: " << adj_small_128_num * sizeof(int)*3.0 / ((t2 - t1) * 1e9) << " GB/s" << endl;

    t1 = omp_get_wtime();
    for(int i = 0; i < small_128_degree_num; i++)
    {
        int src_id = small_128_degree_ids[i];
        long long first = vertex_pointers[src_id];
        long long last = vertex_pointers[src_id + 1];

        #pragma _NEC cncall
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vob
        #pragma _NEC vector
        #pragma _NEC gather_reorder
        for(int i = 0; i < 256; i++)
        {
            long long edge_pos = first + i;
            if(edge_pos < last)
            {
                int dst_id = adjacent_ids[edge_pos];
                result[edge_pos] = data[dst_id];
            }
        }
    }
    t2 = omp_get_wtime();
    cout << "BW small fixed VL 128 single core: " << adj_small_128_num * sizeof(int)*3.0 / ((t2 - t1) * 1e9) << " GB/s" << endl;

    t1 = omp_get_wtime();
    #pragma omp parallel for
    for(int idx = 0; idx < small_128_degree_num - 256; idx += 256)
    {
        int src_id_reg[256];
        long long first_reg[256];
        long long last_reg[256];

        #pragma _NEC vreg(src_id_reg)
        #pragma _NEC vreg(first_reg)
        #pragma _NEC vreg(last_reg)

        #pragma _NEC ivdep
        for(int i = 0; i < 256; i++)
        {
            src_id_reg[i] = small_128_degree_ids[idx + i];
            first_reg[i] = vertex_pointers[idx + i];
            last_reg[i] = vertex_pointers[idx + i + 1];
        }

        int max_conn = 0;
        for(int i = 0; i < 256; i++)
        {
            int conn = last_reg[i] - first_reg[i];
            if (max_conn < conn)
                max_conn = conn;
        }

        for(int pos = 0; pos < max_conn; pos++)
        {
            #pragma _NEC cncall
            #pragma _NEC ivdep
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vob
            #pragma _NEC vector
            #pragma _NEC gather_reorder
            for(int i = 0; i < 256; i++)
            {
                long long edge_pos = first_reg[i] + pos;
                if(edge_pos < last_reg[i])
                {
                    int dst_id = adjacent_ids[edge_pos];
                    result[edge_pos] = data[dst_id];
                }
            }
        }
    }
    t2 = omp_get_wtime();
    cout << "BW small fixed VL 128: " << adj_small_128_num * sizeof(int)*3.0 / ((t2 - t1) * 1e9) << " GB/s" << endl;

    delete []medium_degree_ids;
    delete []data;
    delete []result;
    delete []small_128_degree_ids;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
