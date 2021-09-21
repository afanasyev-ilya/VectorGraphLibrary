/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CSR_VG_Graph::CSR_VG_Graph(int _vertices_count, long long _edges_count)
{
    this->graph_format = CSR_VG_GRAPH;
    this->supported_direction = USE_SCATTER_ONLY;

    alloc(_vertices_count, _edges_count);

    is_copy = false;

    #ifdef __USE_NEC_SX_AURORA__
    cell_c_vertex_groups_num = 0;
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CSR_VG_Graph::CSR_VG_Graph(const CSR_VG_Graph &_copy)
{
    this->graph_format = _copy.graph_format;
    this->supported_direction = _copy.supported_direction;

    this->vertices_count = _copy.vertices_count;
    this->edges_count = _copy.edges_count;

    this->vertex_pointers = _copy.vertex_pointers;
    this->adjacent_ids = _copy.adjacent_ids;
    this->edges_reorder_indexes = _copy.edges_reorder_indexes;

    this->is_copy = true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CSR_VG_Graph::~CSR_VG_Graph()
{
    free();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSR_VG_Graph::alloc(int _vertices_count, long long _edges_count)
{
    this->vertices_count = _vertices_count;
    this->edges_count = _edges_count;

    MemoryAPI::allocate_array(&vertex_pointers, this->vertices_count + 1);
    MemoryAPI::allocate_array(&adjacent_ids, this->edges_count);

    MemoryAPI::allocate_array(&edges_reorder_indexes, this->edges_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSR_VG_Graph::free()
{
    if(!is_copy)
    {
        MemoryAPI::free_array(vertex_pointers);
        MemoryAPI::free_array(adjacent_ids);

        MemoryAPI::free_array(edges_reorder_indexes);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSR_VG_Graph::resize(int _vertices_count, long long _edges_count)
{
    this->free();
    this->alloc(_vertices_count, _edges_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSR_VG_Graph::save_main_content_to_binary_file(FILE *_graph_file)
{
    int vertices_count = this->vertices_count;
    long long edges_count = this->edges_count;

    fwrite(reinterpret_cast<const char*>(&vertices_count), sizeof(int), 1, _graph_file);
    fwrite(reinterpret_cast<const char*>(&edges_count), sizeof(long long), 1, _graph_file);
    fwrite(reinterpret_cast<const char*>(&(this->graph_format)), sizeof(GraphStorageFormat), 1, _graph_file);

    fwrite(reinterpret_cast<const char*>(vertex_pointers), sizeof(long long), vertices_count + 1, _graph_file);
    fwrite(reinterpret_cast<const char*>(adjacent_ids), sizeof(int), edges_count, _graph_file);
    fwrite(reinterpret_cast<const char*>(edges_reorder_indexes), sizeof(vgl_sort_indexes), edges_count, _graph_file);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSR_VG_Graph::load_main_content_from_binary_file(FILE *_graph_file)
{
    fread(reinterpret_cast<char*>(&this->vertices_count), sizeof(int), 1, _graph_file);
    fread(reinterpret_cast<char*>(&this->edges_count), sizeof(long long), 1, _graph_file);
    fread(reinterpret_cast<char*>(&(this->graph_format)), sizeof(GraphStorageFormat), 1, _graph_file);
    if(this->graph_format != CSR_GRAPH)
    {
        throw "Error in CSR_VG_Graph::load_from_binary_file : graph type in file is not equal to CSR_GRAPH";
    }

    resize(this->vertices_count, this->edges_count);

    fread(reinterpret_cast<char*>(vertex_pointers), sizeof(long long), vertices_count + 1, _graph_file);
    fread(reinterpret_cast<char*>(adjacent_ids), sizeof(int), edges_count, _graph_file);
    fread(reinterpret_cast<char*>(edges_reorder_indexes), sizeof(vgl_sort_indexes), edges_count, _graph_file);

    CSR_VG_Graph::create_vertex_groups();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int CSR_VG_Graph::select_random_nz_vertex()
{
    int attempt_num = 0;
    while(attempt_num < ATTEMPTS_THRESHOLD)
    {
        int vertex_id = rand() % this->vertices_count;
        if(get_connections_count(vertex_id) > 0)
            return vertex_id;
        attempt_num++;
    }
    cout << "Error in VectorCSR_VG_Graph::select_random_vertex: can not select non-zero degree vertex in ATTEMPTS_THRESHOLD attempts" << endl;
    return rand() % this->vertices_count;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
void CSR_VG_Graph::move_to_device()
{
    MemoryAPI::move_array_to_device(vertex_pointers, this->vertices_count + 1);
    MemoryAPI::move_array_to_device(adjacent_ids, this->edges_count);
    MemoryAPI::move_array_to_device(edges_reorder_indexes, this->edges_count);

    for(int i = 0; i < CSR_VERTEX_GROUPS_NUM; i++)
        vertex_groups[i].move_to_device();
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
void CSR_VG_Graph::move_to_host()
{
    MemoryAPI::move_array_to_host(vertex_pointers, this->vertices_count + 1);
    MemoryAPI::move_array_to_host(adjacent_ids, this->edges_count);
    MemoryAPI::move_array_to_host(edges_reorder_indexes, this->edges_count);

    for(int i = 0; i < CSR_VERTEX_GROUPS_NUM; i++)
        vertex_groups[i].move_to_host();
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
