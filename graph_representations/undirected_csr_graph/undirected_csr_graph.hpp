#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

UndirectedCSRGraph::UndirectedCSRGraph(int _vertices_count, long long _edges_count)
{
    this->graph_type = UNDIRECTED_CSR_GRAPH;
    this->supported_direction = USE_SCATTER_ONLY;

    alloc(_vertices_count, _edges_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

UndirectedCSRGraph::~UndirectedCSRGraph()
{
    free();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void UndirectedCSRGraph::alloc(int _vertices_count, long long _edges_count)
{
    this->vertices_count = _vertices_count;
    this->edges_count = _edges_count;

    MemoryAPI::allocate_array(&vertex_pointers, this->vertices_count + 1);
    MemoryAPI::allocate_array(&adjacent_ids, this->edges_count);

    MemoryAPI::allocate_array(&forward_conversion, this->vertices_count);
    MemoryAPI::allocate_array(&backward_conversion, this->vertices_count);

    MemoryAPI::allocate_array(&edges_reorder_indexes, this->edges_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void UndirectedCSRGraph::free()
{
    MemoryAPI::free_array(vertex_pointers);
    MemoryAPI::free_array(adjacent_ids);

    MemoryAPI::free_array(forward_conversion);
    MemoryAPI::free_array(backward_conversion);

    MemoryAPI::free_array(edges_reorder_indexes);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void UndirectedCSRGraph::resize(int _vertices_count, long long _edges_count)
{
    this->free();
    this->alloc(_vertices_count, _edges_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue>
void UndirectedCSRGraph::save_to_graphviz_file(string _file_name, VerticesArray<_TVertexValue> &_vertex_data,
                                               VisualisationMode _visualisation_mode)
{
    ofstream dot_output(_file_name.c_str());
    
    string edge_symbol;
    if(_visualisation_mode == VISUALISE_AS_DIRECTED)
    {
        dot_output << "digraph G {" << endl;
        edge_symbol = " -> ";
    }
    else if(_visualisation_mode == VISUALISE_AS_UNDIRECTED)
    {
        dot_output << "graph G {" << endl;
        edge_symbol = " -- ";
    }

    for(int cur_vertex = 0; cur_vertex < this->vertices_count; cur_vertex++)
    {
        dot_output << cur_vertex << " [label = \" " << _vertex_data[cur_vertex] << " \"];" << endl;
    }
    
    for(int cur_vertex = 0; cur_vertex < this->vertices_count; cur_vertex++)
    {
        int src_id = cur_vertex;
        for(long long edge_pos = vertex_pointers[cur_vertex]; edge_pos < vertex_pointers[cur_vertex + 1]; edge_pos++)
        {
            int dst_id = adjacent_ids[edge_pos];
            dot_output << src_id << edge_symbol << dst_id /*<< " [label = \" " << weight << " \"];"*/ << endl;
        }
    }
    
    dot_output << "}";
    dot_output.close();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int UndirectedCSRGraph::select_random_vertex()
{
    int attempt_num = 0;
    while(attempt_num < ATTEMPTS_THRESHOLD)
    {
        int vertex_id = rand() % this->vertices_count;
        if(get_connections_count(vertex_id) > 0)
            return vertex_id;
        attempt_num++;
    }
    throw "Error in UndirectedCSRGraph::select_random_vertex: can not select non-zero degree vertex in ATTEMPTS_THRESHOLD attempts";
    return -1;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void UndirectedCSRGraph::save_main_content_to_binary_file(FILE *_graph_file)
{
    fwrite(reinterpret_cast<const char*>(vertex_pointers), sizeof(long long), vertices_count + 1, _graph_file);
    fwrite(reinterpret_cast<const char*>(adjacent_ids), sizeof(int), edges_count, _graph_file);

    fwrite(reinterpret_cast<const char*>(forward_conversion), sizeof(int), vertices_count, _graph_file);
    fwrite(reinterpret_cast<const char*>(backward_conversion), sizeof(int), vertices_count, _graph_file);
    fwrite(reinterpret_cast<const char*>(edges_reorder_indexes), sizeof(vgl_sort_indexes), edges_count, _graph_file);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void UndirectedCSRGraph::load_main_content_from_binary_file(FILE *_graph_file)
{
    fread(reinterpret_cast<char*>(vertex_pointers), sizeof(long long), vertices_count + 1, _graph_file);
    fread(reinterpret_cast<char*>(adjacent_ids), sizeof(int), edges_count, _graph_file);

    fread(reinterpret_cast<char*>(forward_conversion), sizeof(int), vertices_count, _graph_file);
    fread(reinterpret_cast<char*>(backward_conversion), sizeof(int), vertices_count, _graph_file);
    fread(reinterpret_cast<char*>(edges_reorder_indexes), sizeof(vgl_sort_indexes), edges_count, _graph_file);

    #if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
    estimate_nec_thresholds();
    last_vertices_ve.init_from_graph(this->vertex_pointers, this->adjacent_ids,
                                     vector_core_threshold_vertex, this->vertices_count);
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool UndirectedCSRGraph::save_to_binary_file(string _file_name)
{
    FILE *graph_file = fopen(_file_name.c_str(), "wb");
    if(graph_file == NULL)
        return false;

    int vertices_count = this->vertices_count;
    long long edges_count = this->edges_count;

    fwrite(reinterpret_cast<const char*>(&(this->graph_type)), sizeof(GraphType), 1, graph_file);
    fwrite(reinterpret_cast<const char*>(&vertices_count), sizeof(int), 1, graph_file);
    fwrite(reinterpret_cast<const char*>(&edges_count), sizeof(long long), 1, graph_file);

    save_main_content_to_binary_file(graph_file);

    fclose(graph_file);
    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool UndirectedCSRGraph::load_from_binary_file(string _file_name)
{
    FILE * graph_file = fopen(_file_name.c_str(), "rb");
    if(graph_file == NULL)
        return false;

    fread(reinterpret_cast<char*>(&(this->graph_type)), sizeof(GraphType), 1, graph_file);
    if(this->graph_type != UNDIRECTED_CSR_GRAPH)
    {
        throw "Error in UndirectedCSRGraph::load_from_binary_file : graph type in file is not equal to UNDIRECTED_CSR_GRAPH";
    }

    fread(reinterpret_cast<char*>(&this->vertices_count), sizeof(int), 1, graph_file);
    fread(reinterpret_cast<char*>(&this->edges_count), sizeof(long long), 1, graph_file);

    resize(this->vertices_count, this->edges_count);

    load_main_content_from_binary_file(graph_file);

    fclose(graph_file);
    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////





