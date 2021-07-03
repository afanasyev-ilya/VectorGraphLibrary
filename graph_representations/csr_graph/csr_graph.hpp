/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CSRGraph::CSRGraph(int _vertices_count, long long _edges_count)
{
    this->graph_type = CSR_GRAPH;
    this->supported_direction = USE_SCATTER_ONLY;

    alloc(_vertices_count, _edges_count);
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
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSRGraph::free()
{
    MemoryAPI::free_array(vertex_pointers);
    MemoryAPI::free_array(adjacent_ids);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSRGraph::resize(int _vertices_count, long long _edges_count)
{
    this->free();
    this->alloc(_vertices_count, _edges_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool CSRGraph::save_to_binary_file(string _file_name)
{
    FILE *graph_file = fopen(_file_name.c_str(), "wb");
    if(graph_file == NULL)
        return false;

    int vertices_count = this->vertices_count;
    long long edges_count = this->edges_count;

    fwrite(reinterpret_cast<const char*>(&(this->graph_type)), sizeof(GraphType), 1, graph_file);
    fwrite(reinterpret_cast<const char*>(&vertices_count), sizeof(int), 1, graph_file);
    fwrite(reinterpret_cast<const char*>(&edges_count), sizeof(long long), 1, graph_file);

    fwrite(reinterpret_cast<const char*>(vertex_pointers), sizeof(long long), vertices_count + 1, graph_file);
    fwrite(reinterpret_cast<const char*>(adjacent_ids), sizeof(int), edges_count, graph_file);

    fclose(graph_file);
    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool CSRGraph::load_from_binary_file(string _file_name)
{
    FILE * graph_file = fopen(_file_name.c_str(), "rb");
    if(graph_file == NULL)
        return false;

    fread(reinterpret_cast<char*>(&(this->graph_type)), sizeof(GraphType), 1, graph_file);
    if(this->graph_type != CSR_GRAPH)
    {
        throw "Error in CSRGraph::load_from_binary_file : graph type in file is not equal to CSR_GRAPH";
    }

    fread(reinterpret_cast<char*>(&this->vertices_count), sizeof(int), 1, graph_file);
    fread(reinterpret_cast<char*>(&this->edges_count), sizeof(long long), 1, graph_file);

    resize(this->vertices_count, this->edges_count);

    fread(reinterpret_cast<char*>(vertex_pointers), sizeof(long long), vertices_count + 1, graph_file);
    fread(reinterpret_cast<char*>(adjacent_ids), sizeof(int), edges_count, graph_file);

    fclose(graph_file);
    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
