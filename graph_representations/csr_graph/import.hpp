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

void CSRGraph::create_vertices_group_array(CSRVertexGroup &_group_data, int _bottom, int _top)
{
    int local_group_size = 0;
    long long local_group_neighbours = 0;

    for(int src_id = 0; src_id < this->vertices_count; src_id++)
    {
        int connections_count = vertex_pointers[src_id + 1] - vertex_pointers[src_id];
        if((connections_count >= _bottom) && (connections_count < _top))
        {
            local_group_neighbours += connections_count;
            local_group_size++;
        }
    }

    _group_data.resize(local_group_size);
    _group_data.neighbours = local_group_neighbours;

    int pos = 0;
    for(int src_id = 0; src_id < this->vertices_count; src_id++)
    {
        int connections_count = vertex_pointers[src_id + 1] - vertex_pointers[src_id];
        if((connections_count >= _bottom) && (connections_count < _top))
        {
            _group_data.ids[pos] = src_id;
            pos++;
        }
    }

    /*cout << "borders: " << _bottom << " " << _top << endl;
    cout << "size: " << _group_data.size << endl;
    cout << "neighbours: " << _group_data.neighbours << endl;*/
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSRGraph::test_advance_changed_vl(CSRVertexGroup &_group_data, string _name)
{
    double t1 = omp_get_wtime();
    #pragma omp parallel for schedule(static, 8)
    for(int idx = 0; idx < _group_data.size; idx++)
    {
        int src_id = _group_data.ids[idx];
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
    cout << _name + " changed VL: " <<  _group_data.neighbours * sizeof(int)*3.0 / ((t2 - t1) * 1e9) << " GB/s" << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSRGraph::test_advance_virtual_warp(CSRVertexGroup &_group_data, string _name)
{
    double t1 = omp_get_wtime();
    #pragma omp parallel
    {
        long long edge_pos_reg[VECTOR_LENGTH];
        long long last_reg[VECTOR_LENGTH];

        #pragma _NEC vreg(edge_pos_reg)
        #pragma _NEC vreg(last_reg)

        #pragma _NEC vector
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            edge_pos_reg[i] = 0;
            last_reg[i] = 0;
        }

        #pragma omp for schedule(static, 8)
        for(int idx = 0; idx < _group_data.size - 2; idx += 2)
        {
            const int src_ids[2] = {_group_data.ids[idx], _group_data.ids[idx + 1]};
            const long long firsts[2] = {vertex_pointers[src_ids[0]], vertex_pointers[src_ids[1]]};
            const long long lasts[2] = {vertex_pointers[src_ids[0] + 1], vertex_pointers[src_ids[1] + 1]};

            #pragma _NEC cncall
            #pragma _NEC ivdep
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vob
            #pragma _NEC vector
            #pragma _NEC gather_reorder
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                int warp_num = 0;//i / 128;
                int warp_pos = i;//i % 128;
                edge_pos_reg[i] = firsts[warp_num] + warp_pos;
                last_reg[i] = lasts[warp_num];
            }

            #pragma _NEC cncall
            #pragma _NEC ivdep
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vob
            #pragma _NEC vector
            #pragma _NEC gather_reorder
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                long long edge_pos = edge_pos_reg[i];
                if(edge_pos < last_reg[i])
                {
                    int dst_id = adjacent_ids[edge_pos];
                    result[edge_pos] = data[dst_id];
                }
            }
        }
    }
    double t2 = omp_get_wtime();
    cout << _name + " virtual warp: " <<  _group_data.neighbours * sizeof(int)*3.0 / ((t2 - t1) * 1e9) << " GB/s" << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSRGraph::test_advance_fixed_vl(CSRVertexGroup &_group_data, string _name)
{
    double t1 = omp_get_wtime();
    #pragma omp parallel for schedule(static, 8)
    for(int idx = 0; idx < _group_data.size; idx++)
    {
        int src_id = _group_data.ids[idx];
        long long first = vertex_pointers[src_id];
        long long last = vertex_pointers[src_id + 1];

        #pragma _NEC cncall
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vob
        #pragma _NEC vector
        #pragma _NEC gather_reorder
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            long long edge_pos = first + i;
            if(edge_pos < last)
            {
                int dst_id = adjacent_ids[edge_pos];
                result[edge_pos] = data[dst_id];
            }
        }
    }
    double t2 = omp_get_wtime();
    cout << _name + " fixed VL: " <<  _group_data.neighbours * sizeof(int)*3.0 / ((t2 - t1) * 1e9) << " GB/s" << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSRGraph::test_advance_sparse(CSRVertexGroup &_group_data, string _name)
{
    double t1 = omp_get_wtime();
    #pragma omp parallel
    {
        int src_id_reg[VECTOR_LENGTH];
        long long first_reg[VECTOR_LENGTH];
        long long last_reg[VECTOR_LENGTH];
        int connections_reg[VECTOR_LENGTH];
        int res_reg[VECTOR_LENGTH];

        #pragma _NEC vreg(src_id_reg)
        #pragma _NEC vreg(first_reg)
        #pragma _NEC vreg(last_reg)
        #pragma _NEC vreg(connections_reg)
        #pragma _NEC vreg(res_reg)

        #pragma _NEC vector
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            src_id_reg[i] = 0;
            first_reg[i] = 0;
            last_reg[i] = 0;
            connections_reg[i] = 0;
            res_reg[i] = 0;
        }

        #pragma omp for schedule(static, 8)
        for(int idx = 0; idx < _group_data.size; idx += VECTOR_LENGTH)
        {
            #pragma _NEC ivdep
            #pragma _NEC vector
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                if((idx + i) < _group_data.size)
                {
                    int src_id = _group_data.ids[idx + i];
                    src_id_reg[i] = src_id;
                    first_reg[i] = vertex_pointers[src_id];
                    last_reg[i] = vertex_pointers[src_id + 1];
                    connections_reg[i] = last_reg[i] - first_reg[i];
                }
            }

            int max_conn = 0;
            #pragma _NEC vector
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                int conn = connections_reg[i];
                if (((idx + i) < _group_data.size) && (max_conn < conn))
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
                for(int i = 0; i < VECTOR_LENGTH; i++)
                {
                    long long edge_pos = first_reg[i] + pos;
                    if(((idx + i) < _group_data.size) && (edge_pos < last_reg[i]))
                    {
                        int dst_id = adjacent_ids[edge_pos];
                        result[edge_pos] = data[dst_id];
                    }
                }
            }
        }
    }
    double t2 = omp_get_wtime();
    cout << _name + " sparse: " << _group_data.neighbours * sizeof(int)*3.0 / ((t2 - t1) * 1e9) << " GB/s" << endl;
}

void CSRGraph::test_advance_sparse_packed(CSRVertexGroup &_group_data, string _name)
{
    long long *packed_ids = (long long*)adjacent_ids;
    /*long long *long_ptrs = new long long[this->vertices_count + 1];
    #pragma omp parallel for
    for(int i = 0; i < this->vertices_count + 1; i++)
    {
        long_ptrs[i] = vertex_pointers[i] / 2;
    }*/

    double t1 = omp_get_wtime();
    #pragma omp parallel
    {
        int src_id_reg[VECTOR_LENGTH];
        long long first_reg[VECTOR_LENGTH];
        long long last_reg[VECTOR_LENGTH];
        int connections_reg[VECTOR_LENGTH];
        long long res_reg[VECTOR_LENGTH];

        #pragma _NEC vreg(src_id_reg)
        #pragma _NEC vreg(first_reg)
        #pragma _NEC vreg(last_reg)
        #pragma _NEC vreg(connections_reg)
        #pragma _NEC vreg(res_reg)

        #pragma _NEC vector
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            src_id_reg[i] = 0;
            first_reg[i] = 0;
            last_reg[i] = 0;
            connections_reg[i] = 0;
            res_reg[i] = 0;
        }

        #pragma omp for schedule(static, 8)
        for(int idx = 0; idx < _group_data.size; idx += VECTOR_LENGTH)
        {
            #pragma _NEC ivdep
            #pragma _NEC vector
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                if((idx + i) < _group_data.size)
                {
                    int src_id = _group_data.ids[idx + i];
                    src_id_reg[i] = src_id;
                    first_reg[i] = vertex_pointers[src_id] / 2;
                    last_reg[i] = vertex_pointers[src_id + 1] / 2;
                    connections_reg[i] = last_reg[i] - first_reg[i];
                }
            }

            int max_conn = 0;
            #pragma _NEC vector
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                int conn = connections_reg[i];
                if (((idx + i) < _group_data.size) && (max_conn < conn))
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
                for(int i = 0; i < VECTOR_LENGTH; i++)
                {
                    long long edge_pos = first_reg[i] + pos;
                    long long write_pos = first_reg[i]*2 + pos*2;
                    if(((idx + i) < _group_data.size) && (edge_pos < last_reg[i]))
                    {
                        long long packed_id = packed_ids[edge_pos];
                        int first_id = (int)((packed_id & 0xFFFFFFFF00000000LL) >> 32);
                        int second_id = (int)(packed_id & 0xFFFFFFFFLL);
                        int val_1 = data[first_id];
                        int val_2 = data[second_id];
                        res_reg[i] += val_1 + val_2;
                    }

                    /*if((write_pos + 1) < this->edges_count)
                    {
                        //int first_id = 0, second_id = 0;
                        //vgl_unpack(packed_id, first_id, second_id);
                        int first_id = (int)((packed_id & 0xFFFFFFFF00000000LL) >> 32);
                        int second_id = (int)(packed_id & 0xFFFFFFFFLL);

                        data[second_id] = data[first_id];
                    }*/
                }
            }
        }
    }
    double t2 = omp_get_wtime();
    cout << _name + " sparse packed 2: " << _group_data.neighbours * sizeof(int)*3.0 / ((t2 - t1) * 1e9) << " GB/s" << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSRGraph::import(EdgesListGraph &_el_graph, bool _random_shuffle_required)
{
    resize(_el_graph.get_vertices_count(), _el_graph.get_edges_count());
    construct_unsorted_csr(_el_graph, _random_shuffle_required);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSRGraph::test_advance()
{
    CSRVertexGroup large_degree, medium_degree, small_degree, lowest_degree, test_group;

    create_vertices_group_array(large_degree, VECTOR_LENGTH, 2147483647);
    create_vertices_group_array(test_group, 32, 256);
    create_vertices_group_array(medium_degree, 64, 128);
    create_vertices_group_array(small_degree, 16, 64);
    create_vertices_group_array(lowest_degree, 0, 16);

    // test advance
    data = new int[this->vertices_count];
    result = new int[this->edges_count];

    #pragma omp parallel
    {};

    test_advance_changed_vl(large_degree, "VECTOR_LENGTH - inf");

    cout << "-----testing medium region-------" << endl;
    test_advance_changed_vl(medium_degree, "64 - 128");
    test_advance_fixed_vl(medium_degree, "64 - 128");
    test_advance_sparse(medium_degree, " 64 - 128");
    //test_advance_sparse_packed(medium_degree, " 64 - 128");
    //test_advance_virtual_warp(medium_degree, " 64 - 128");
    cout << "-----testing medium region-------" << endl;

    cout << "-----testing small region-------" << endl;
    test_advance_fixed_vl(small_degree, "32 - VECTOR_LENGTH");
    test_advance_sparse(small_degree, " 16 - 32");
    test_advance_sparse(lowest_degree, " 0 - 16");
    cout << "-----testing small region-------" << endl;

    cout << "-----TEST-------" << endl;
    test_advance_changed_vl(test_group, " test group 32 - 256");
    test_advance_fixed_vl(medium_degree, " test group 32 - 256");
    test_advance_sparse(test_group, " test group 32 - 256");
    //test_advance_sparse_packed(test_group, " test group 32 - 256");
    cout << "-----TEST-------" << endl;

    delete []data;
    delete []result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSRGraph::test_full_advance()
{
    // test advance
    data = new int[this->vertices_count];
    result = new int[this->edges_count];

    #pragma omp parallel
    {};

    CSRVertexGroup large_degree;
    CSRVertexGroup degree_256_to_128;
    CSRVertexGroup degree_128_to_64;
    CSRVertexGroup degree_64_to_32;
    CSRVertexGroup degree_32_to_16;
    CSRVertexGroup degree_16_to_8;
    CSRVertexGroup degree_8_to_0;
    create_vertices_group_array(large_degree, 256, 2147483647);
    create_vertices_group_array(degree_256_to_128, 128, 256);
    create_vertices_group_array(degree_128_to_64, 64, 128);
    create_vertices_group_array(degree_64_to_32, 32, 64);
    create_vertices_group_array(degree_32_to_16, 16, 32);
    create_vertices_group_array(degree_16_to_8, 8, 16);
    create_vertices_group_array(degree_8_to_0, 0, 8);

    test_advance_changed_vl(large_degree, " large_degree");
    test_advance_fixed_vl(degree_256_to_128, " fixed degree_256_to_128");
    test_advance_changed_vl(degree_256_to_128, " changed degree_256_to_128");
    test_advance_sparse(degree_128_to_64, " fixed degree_128_to_64");
    test_advance_changed_vl(degree_128_to_64, " changed degree_128_to_64");
    test_advance_sparse(degree_64_to_32, " degree_64_to_32");
    test_advance_sparse(degree_32_to_16, " degree_32_to_16");
    test_advance_sparse(degree_16_to_8, " degree_16_to_8");
    test_advance_sparse(degree_8_to_0, " degree_8_to_0");

    delete []data;
    delete []result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
