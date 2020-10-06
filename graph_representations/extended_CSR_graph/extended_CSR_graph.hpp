#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ExtendedCSRGraph::ExtendedCSRGraph(int _vertices_count, long long _edges_count)
{
    this->graph_type = GraphTypeExtendedCSR;
    alloc(_vertices_count, _edges_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ExtendedCSRGraph::~ExtendedCSRGraph()
{
    free();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ExtendedCSRGraph::alloc(int _vertices_count, long long _edges_count)
{
    this->vertices_count = _vertices_count;
    this->edges_count = _edges_count;

    MemoryAPI::allocate_array(&vertex_pointers, this->vertices_count + 1);
    MemoryAPI::allocate_array(&adjacent_ids, this->edges_count);

    MemoryAPI::allocate_array(&forward_conversion, this->vertices_count);
    MemoryAPI::allocate_array(&backward_conversion, this->vertices_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ExtendedCSRGraph::free()
{
    MemoryAPI::free_array(vertex_pointers);
    MemoryAPI::free_array(adjacent_ids);

    MemoryAPI::free_array(forward_conversion);
    MemoryAPI::free_array(backward_conversion);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ExtendedCSRGraph::resize(int _vertices_count, long long _edges_count)
{
    this->free();
    this->alloc(_vertices_count, _edges_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ExtendedCSRGraph::save_to_graphviz_file(string _file_name, VisualisationMode _visualisation_mode)
{
    ofstream dot_output(_file_name.c_str());
    
    string connection;
    if(_visualisation_mode == VISUALISE_AS_DIRECTED)
    {
        dot_output << "digraph G {" << endl;
        connection = " -> ";
    }
    else if(_visualisation_mode == VISUALISE_AS_UNDIRECTED)
    {
        dot_output << "graph G {" << endl;
        connection = " -- ";
    }
    
    for(int cur_vertex = 0; cur_vertex < this->vertices_count; cur_vertex++)
    {
        int src_id = cur_vertex;
        for(long long edge_pos = vertex_pointers[cur_vertex]; edge_pos < vertex_pointers[cur_vertex + 1]; edge_pos++)
        {
            int dst_id = adjacent_ids[edge_pos];
            float weight = 0.0;
            if(src_id != dst_id)
            {
                if(_visualisation_mode == VISUALISE_AS_UNDIRECTED)
                {
                    if(src_id < dst_id)
                    {
                        dot_output << src_id << connection << dst_id << " [label = \" " << weight << " \"];" << endl;
                    }
                }
                else
                {
                    dot_output << src_id << connection << dst_id << " [label = \" " << weight << " \"];" << endl;
                }
            }
        }
    }
    
    dot_output << "}";
    dot_output.close();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool ExtendedCSRGraph::save_to_binary_file(string _file_name)
{
    FILE * graph_file = fopen(_file_name.c_str(), "wb");
    if(graph_file == NULL)
        return false;
    
    int vertices_count = this->vertices_count;
    long long edges_count = this->edges_count;
    /*fwrite(reinterpret_cast<const void*>(&(this->graph_type)), sizeof(GraphType), 1, graph_file);
    fwrite(reinterpret_cast<const void*>(&vertices_count), sizeof(int), 1, graph_file);
    fwrite(reinterpret_cast<const void*>(&edges_count), sizeof(long long), 1, graph_file);

    //fwrite(reinterpret_cast<const char*>(reordered_vertex_ids), sizeof(int), vertices_count, graph_file);
    fwrite(reinterpret_cast<const char*>(vertex_pointers), sizeof(long long), vertices_count + 1, graph_file);
    
    fwrite(reinterpret_cast<const char*>(adjacent_ids), sizeof(int), edges_count, graph_file);*/
    
    fclose(graph_file);
    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


bool ExtendedCSRGraph::load_from_binary_file(string _file_name)
{
    FILE * graph_file = fopen(_file_name.c_str(), "rb");
    if(graph_file == NULL)
        return false;
    
    /*fread(reinterpret_cast<void*>(&this->vertices_count), sizeof(int), 1, graph_file);
    fread(reinterpret_cast<void*>(&this->edges_count), sizeof(long long), 1, graph_file);
    
    this->resize(this->vertices_count, this->edges_count);
    
    fread(reinterpret_cast<void*>(&(this->graph_type)), sizeof(GraphType), 1, graph_file);
    
    if(this->graph_type != GraphTypeExtendedCSR)
        throw "ERROR: loaded incorrect type of graph into ExtendedCSRGraph container";
    
    fread(reinterpret_cast<void*>(&vertices_state), sizeof(VerticesState), 1, graph_file);
    fread(reinterpret_cast<void*>(&edges_state), sizeof(EdgesState), 1, graph_file);
    fread(reinterpret_cast<void*>(&supported_vector_length), sizeof(int), 1, graph_file);

    //fread(reinterpret_cast<char*>(reordered_vertex_ids), sizeof(int), this->vertices_count, graph_file);
    fread(reinterpret_cast<char*>(vertex_pointers), sizeof(long long), (this->vertices_count + 1), graph_file);
    
    fread(reinterpret_cast<char*>(adjacent_ids), sizeof(int), this->edges_count, graph_file);

    #ifdef __USE_GPU__
    estimate_gpu_thresholds();
    #endif

    #ifdef __USE_NEC_SX_AURORA__
    double t1 = omp_get_wtime();
    estimate_nec_thresholds();
    last_vertices_ve.init_from_graph(this->vertex_pointers, this->adjacent_ids, this->adjacent_weights,
                                     vector_core_threshold_vertex, this->vertices_count);
    double t2 = omp_get_wtime();
    cout << "NEC preprocess time: " << t2 - t1 << " sec" << endl;
    #endif*/
    
    fclose(graph_file);
    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

size_t ExtendedCSRGraph::get_graph_size_in_bytes()
{
    size_t graph_size = 0;
    graph_size += this->vertices_count * sizeof(int);
    graph_size += (this->vertices_count + 1) * sizeof(long long);
    graph_size += this->edges_count * sizeof(int);
    return graph_size;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
_T& ExtendedCSRGraph::get_edge_data(_T *_data_array, int _src_id, int _dst_id)
{
    const long long edge_start = vertex_pointers[_src_id];
    const int connections_count = vertex_pointers[_src_id + 1] - vertex_pointers[_src_id];

    for (int edge_pos = 0; edge_pos < connections_count; edge_pos++)
    {
        long long int global_edge_pos = edge_start + edge_pos;
        int current_dst_id = adjacent_ids[global_edge_pos];

        if (_dst_id == current_dst_id)
            return _data_array[global_edge_pos];
    }

    return _data_array[edge_start];
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

long long ExtendedCSRGraph::get_csr_edge_id(int _src_id, int _dst_id)
{
    const long long int start = vertex_pointers[_src_id];
    const long long int end = vertex_pointers[_src_id + 1];
    const int connections_count = end - start;

    for (int local_edge_pos = 0; local_edge_pos < connections_count; local_edge_pos++)
    {
        const long long global_edge_pos = start + local_edge_pos;
        const int dst_id = adjacent_ids[global_edge_pos];
        if(dst_id == _dst_id)
        {
            return global_edge_pos;
        }
    }
    throw "Error in ExtendedCSRGraph::get_csr_edge_id(): specified dst_id not found for current src vertex";
    return -1;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


