/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CSRGraph::CSRGraph(int _vertices_count, long long _edges_count)
{
    this->get_format = CSR_GRAPH;
    this->supported_direction = USE_SCATTER_ONLY;

    alloc(_vertices_count, _edges_count);

    is_copy = false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CSRGraph::CSRGraph(const CSRGraph &_copy)
{
    this->get_format = _copy.get_format;
    this->supported_direction = _copy.supported_direction;

    this->vertices_count = _copy.vertices_count;
    this->edges_count = _copy.edges_count;

    this->vertex_pointers = _copy.vertex_pointers;
    this->adjacent_ids = _copy.adjacent_ids;
    this->edges_reorder_indexes = _copy.edges_reorder_indexes;

    this->is_copy = true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CSRGraph::~CSRGraph()
{
    free();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSRGraph::alloc(int _vertices_count, long long _edges_count)
{
    this->vertices_count = _vertices_count;
    this->edges_count = _edges_count;

    MemoryAPI::allocate_array(&vertex_pointers, this->vertices_count + 1);
    MemoryAPI::allocate_array(&adjacent_ids, this->edges_count);

    MemoryAPI::allocate_array(&edges_reorder_indexes, this->edges_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSRGraph::free()
{
    if(!is_copy)
    {
        MemoryAPI::free_array(vertex_pointers);
        MemoryAPI::free_array(adjacent_ids);

        MemoryAPI::free_array(edges_reorder_indexes);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSRGraph::resize(int _vertices_count, long long _edges_count)
{
    this->free();
    this->alloc(_vertices_count, _edges_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSRGraph::save_main_content_to_binary_file(FILE *_graph_file)
{
    int vertices_count = this->vertices_count;
    long long edges_count = this->edges_count;

    fwrite(reinterpret_cast<const char*>(&vertices_count), sizeof(int), 1, _graph_file);
    fwrite(reinterpret_cast<const char*>(&edges_count), sizeof(long long), 1, _graph_file);
    fwrite(reinterpret_cast<const char*>(&(this->get_format)), sizeof(GraphFormatType), 1, _graph_file);

    fwrite(reinterpret_cast<const char*>(vertex_pointers), sizeof(long long), vertices_count + 1, _graph_file);
    fwrite(reinterpret_cast<const char*>(adjacent_ids), sizeof(int), edges_count, _graph_file);
    fwrite(reinterpret_cast<const char*>(edges_reorder_indexes), sizeof(vgl_sort_indexes), edges_count, _graph_file);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSRGraph::load_main_content_from_binary_file(FILE *_graph_file)
{
    fread(reinterpret_cast<char*>(&this->vertices_count), sizeof(int), 1, _graph_file);
    fread(reinterpret_cast<char*>(&this->edges_count), sizeof(long long), 1, _graph_file);
    fread(reinterpret_cast<char*>(&(this->get_format)), sizeof(GraphFormatType), 1, _graph_file);
    if(this->get_format != CSR_GRAPH)
    {
        throw "Error in CSRGraph::load_from_binary_file : graph type in file is not equal to CSR_GRAPH";
    }

    resize(this->vertices_count, this->edges_count);

    fread(reinterpret_cast<char*>(vertex_pointers), sizeof(long long), vertices_count + 1, _graph_file);
    fread(reinterpret_cast<char*>(adjacent_ids), sizeof(int), edges_count, _graph_file);
    fread(reinterpret_cast<char*>(edges_reorder_indexes), sizeof(vgl_sort_indexes), edges_count, _graph_file);

    #ifdef __USE_CSR_VERTEX_GROUPS__
    CSRGraph::create_vertex_groups();
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int CSRGraph::select_random_nz_vertex()
{
    int attempt_num = 0;
    while(attempt_num < ATTEMPTS_THRESHOLD)
    {
        int vertex_id = rand() % this->vertices_count;
        if(get_connections_count(vertex_id) > 0)
            return vertex_id;
        attempt_num++;
    }
    throw "Error in VectorCSRGraph::select_random_vertex: can not select non-zero degree vertex in ATTEMPTS_THRESHOLD attempts";
    return -1;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
void CSRGraph::move_to_device()
{
    if(this->graph_on_device)
    {
        return;
    }

    this->graph_on_device = true;

    MemoryAPI::move_array_to_device(vertex_pointers, this->vertices_count + 1);
    MemoryAPI::move_array_to_device(adjacent_ids, this->edges_count);
    MemoryAPI::move_array_to_device(edges_reorder_indexes, this->edges_count);

    #ifdef __USE_CSR_VERTEX_GROUPS__
    for(int i = 0; i < CSR_VERTEX_GROUPS_NUM; i++)
        vertex_groups[i].move_to_device();
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
void CSRGraph::move_to_host()
{
    if(!this->graph_on_device)
    {
        return;
    }

    this->graph_on_device = false;

    MemoryAPI::move_array_to_host(vertex_pointers, this->vertices_count + 1);
    MemoryAPI::move_array_to_host(adjacent_ids, this->edges_count);
    MemoryAPI::move_array_to_host(edges_reorder_indexes, this->edges_count);

    #ifdef __USE_CSR_VERTEX_GROUPS__
    for(int i = 0; i < CSR_VERTEX_GROUPS_NUM; i++)
        vertex_groups[i].move_to_host();
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
