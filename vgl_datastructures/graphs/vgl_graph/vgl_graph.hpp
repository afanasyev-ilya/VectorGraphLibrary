/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

VGL_Graph::VGL_Graph(GraphStorageFormat _container_type, GraphStorageOptimizations _optimizations)
{
    graph_format = VGL_GRAPH;

    create_containers(_container_type, _optimizations);

    MemoryAPI::allocate_array(&vertices_reorder_buffer, 1);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

VGL_Graph::~VGL_Graph()
{
    delete outgoing_data;
    delete incoming_data;
    MemoryAPI::free_array(vertices_reorder_buffer);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VGL_Graph::create_containers(GraphStorageFormat _container_type, GraphStorageOptimizations _optimizations)
{
    if(_container_type == VECTOR_CSR_GRAPH)
    {
        outgoing_data = new VectorCSRGraph();
        incoming_data = new VectorCSRGraph();
    }
    else if(_container_type == EDGES_LIST_GRAPH)
    {
        outgoing_data = new EdgesListGraph(_optimizations);
        incoming_data = new EdgesListGraph(_optimizations);
    }
    else if(_container_type == CSR_GRAPH)
    {
        outgoing_data = new CSRGraph();
        incoming_data = new CSRGraph();
    }
    else if(_container_type == CSR_VG_GRAPH)
    {
        outgoing_data = new CSR_VG_Graph();
        incoming_data = new CSR_VG_Graph();
    }
    else
    {
        throw "Error: unsupported graph type in VGL_Graph::VGL_Graph";
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VGL_Graph::import(EdgesContainer &_edges_container)
{
    this->vertices_count = _edges_container.get_vertices_count();
    this->edges_count = _edges_container.get_edges_count();
    outgoing_data->import(_edges_container);
    _edges_container.transpose();
    incoming_data->import(_edges_container);
    _edges_container.transpose();

    MemoryAPI::free_array(vertices_reorder_buffer);
    MemoryAPI::allocate_array(&vertices_reorder_buffer, this->vertices_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define PRINT_VERTICES_THRESHOLD 32
#define PRINT_EDGES_THRESHOLD PRINT_VERTICES_THRESHOLD * 4

void VGL_Graph::print()
{
    if((this->vertices_count < PRINT_VERTICES_THRESHOLD) && (this->edges_count < PRINT_EDGES_THRESHOLD))
    {
        outgoing_data->print();
        incoming_data->print();
    }
    else
    {
        cout << "Warning! Graph is too large to print: " << this->vertices_count << " vertices, "
            << this->edges_count << " edges" << endl;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VGL_Graph::print_size()
{
    outgoing_data->print_size();
    incoming_data->print_size();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

UndirectedGraph *VGL_Graph::get_direction_data(TraversalDirection _direction)
{
    if(_direction == SCATTER)
        return outgoing_data;
    else
        return incoming_data;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool VGL_Graph::save_to_binary_file(string _file_name)
{
    FILE *graph_file = fopen(_file_name.c_str(), "wb");
    if(graph_file == NULL)
        return false;

    int vertices_count = this->vertices_count;
    long long edges_count = this->edges_count;
    GraphStorageFormat container_type = get_container_type();

    fwrite(reinterpret_cast<const char*>(&vertices_count), sizeof(int), 1, graph_file);
    fwrite(reinterpret_cast<const char*>(&edges_count), sizeof(long long), 1, graph_file);
    fwrite(reinterpret_cast<const char*>(&container_type), sizeof(GraphStorageFormat), 1, graph_file);

    outgoing_data->save_main_content_to_binary_file(graph_file);
    incoming_data->save_main_content_to_binary_file(graph_file);

    fclose(graph_file);
    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool VGL_Graph::load_from_binary_file(string _file_name)
{
    FILE * graph_file = fopen(_file_name.c_str(), "rb");
    if(graph_file == NULL)
        return false;

    GraphStorageFormat new_container_type = VGL_GRAPH;
    fread(reinterpret_cast<char*>(&this->vertices_count), sizeof(int), 1, graph_file);
    fread(reinterpret_cast<char*>(&this->edges_count), sizeof(long long), 1, graph_file);
    fread(reinterpret_cast<char*>(&new_container_type), sizeof(GraphStorageFormat), 1, graph_file);

    if(new_container_type != get_container_type())
    {
        cout << "Warning! changing container type from " << get_graph_format_name(get_container_type()) << " to "
                << get_graph_format_name(new_container_type) << endl;
    }

    delete outgoing_data;
    delete incoming_data;
    MemoryAPI::free_array(vertices_reorder_buffer);

    create_containers(new_container_type, OPT_NONE);

    MemoryAPI::allocate_array(&vertices_reorder_buffer, this->vertices_count);
    outgoing_data->load_main_content_from_binary_file(graph_file);
    incoming_data->load_main_content_from_binary_file(graph_file);

    fclose(graph_file);
    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int VGL_Graph::select_random_nz_vertex(TraversalDirection _direction)
{
    if(_direction == SCATTER)
        return outgoing_data->select_random_nz_vertex();
    else if(_direction == GATHER)
        return incoming_data->select_random_nz_vertex();
    return reorder(outgoing_data->select_random_nz_vertex(), SCATTER, ORIGINAL); // TODO
    //return reorder(outgoing_data->select_random_vertex(), SCATTER, _direction);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
void VGL_Graph::move_to_device()
{
    outgoing_data->move_to_device();
    if(get_number_of_directions() == BOTH_DIRECTIONS)
        incoming_data->move_to_device();
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
void VGL_Graph::move_to_host()
{
    outgoing_data->move_to_host();
    if(get_number_of_directions() == BOTH_DIRECTIONS)
        incoming_data->move_to_host();
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////





