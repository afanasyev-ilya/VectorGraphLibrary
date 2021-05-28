#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

VectCSRGraph::VectCSRGraph(SupportedDirection _supported_direction,
                           int _vertices_count,
                           long long _edges_count)
{
    this->graph_type = VECT_CSR_GRAPH;
    this->supported_direction = _supported_direction;

    init(_vertices_count, _edges_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

VectCSRGraph::~VectCSRGraph()
{
    free();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VectCSRGraph::init(int _vertices_count, long long _edges_count)
{
    this->vertices_count = _vertices_count;
    this->edges_count = _edges_count;

    outgoing_graph = NULL;
    incoming_graph = NULL;

    if(outgoing_is_stored())
        outgoing_graph = new UndirectedCSRGraph(this->vertices_count, this->edges_count );

    if(incoming_is_stored())
        incoming_graph = new UndirectedCSRGraph(this->vertices_count, this->edges_count );

    MemoryAPI::allocate_array(&vertices_reorder_buffer, this->vertices_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VectCSRGraph::free()
{
    if(outgoing_is_stored())
        delete outgoing_graph;

    if(incoming_is_stored())
        delete incoming_graph;

    MemoryAPI::free_array(vertices_reorder_buffer);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VectCSRGraph::resize(int _vertices_count, long long _edges_count)
{
    free();
    init(_vertices_count, _edges_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue>
void VectCSRGraph::save_to_graphviz_file(string _file_name, VerticesArray<_TVertexValue> &_vertex_data)
{
    // if undirected - take from TODO variable
    if(outgoing_is_stored())
        outgoing_graph->save_to_graphviz_file(_file_name, _vertex_data, VISUALISE_AS_DIRECTED);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int VectCSRGraph::select_random_vertex(TraversalDirection _direction)
{
    if(outgoing_is_stored())
    {
        return reorder(outgoing_graph->select_random_vertex(), SCATTER, ORIGINAL);
    }
    else if(incoming_is_stored())
    {
        return reorder(incoming_graph->select_random_vertex(), GATHER, ORIGINAL);
    }

    throw "Error in VectCSRGraph::select_random_vertex: can not select non-zero degree vertex in ATTEMPTS_THRESHOLD attempts";

    return -1;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool VectCSRGraph::save_to_binary_file(string _file_name)
{
    FILE * graph_file = fopen(_file_name.c_str(), "wb");
    if(graph_file == NULL)
        return false;

    if(!incoming_is_stored())
    {
        throw "Error in VectCSRGraph::save_to_binary_file : saved graph must have both directions";
    }

    int vertices_count = this->vertices_count;
    long long edges_count = this->edges_count;
    fwrite(reinterpret_cast<const char*>(&(this->graph_type)), sizeof(GraphType), 1, graph_file);
    fwrite(reinterpret_cast<const char*>(&vertices_count), sizeof(int), 1, graph_file);
    fwrite(reinterpret_cast<const char*>(&edges_count), sizeof(long long), 1, graph_file);

    outgoing_graph->save_main_content_to_binary_file(graph_file);
    incoming_graph->save_main_content_to_binary_file(graph_file);

    fclose(graph_file);
    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool VectCSRGraph::load_from_binary_file(string _file_name)
{
    FILE * graph_file = fopen(_file_name.c_str(), "rb");
    if(graph_file == NULL)
        return false;

    fread(reinterpret_cast<char*>(&(this->graph_type)), sizeof(GraphType), 1, graph_file);
    if(this->graph_type != VECT_CSR_GRAPH)
    {
        throw "Error in VectCSRGraph::load_from_binary_file : graph type in file is not equal to VECT_CSR_GRAPH";
    }

    fread(reinterpret_cast<char*>(&this->vertices_count), sizeof(int), 1, graph_file);
    fread(reinterpret_cast<char*>(&this->edges_count), sizeof(long long), 1, graph_file);

    this->resize(this->vertices_count, this->edges_count);

    if(outgoing_is_stored())
        outgoing_graph->load_main_content_from_binary_file(graph_file);
    else
        incoming_graph->load_main_content_from_binary_file(graph_file); // TODO this should be equal to skip

    if(incoming_is_stored())
        incoming_graph->load_main_content_from_binary_file(graph_file);

    fclose(graph_file);
    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

