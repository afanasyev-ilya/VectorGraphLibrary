/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct CSRVertexGroup
{
    int *ids;
    int size;
    long long neighbours;

    CSRVertexGroup()
    {
        size = 1;
        neighbours = 0;
        MemoryAPI::allocate_array(&ids, size);
    }

    void resize(int _new_size)
    {
        size = _new_size;
        MemoryAPI::free_array(ids);
        MemoryAPI::allocate_array(&ids, size);
    }

    ~CSRVertexGroup()
    {
        MemoryAPI::free_array(ids);
    }
};

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
