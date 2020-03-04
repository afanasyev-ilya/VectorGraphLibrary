#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::ExtendedCSRGraph(int _vertices_count, long long _edges_count)
{
    this->graph_type = GraphTypeExtendedCSR;
    
    supported_vector_length = VECTOR_LENGTH;
    vertices_state = VERTICES_UNSORTED;
    edges_state = EDGES_UNSORTED;
    
    alloc(_vertices_count, _edges_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::~ExtendedCSRGraph()
{
    free();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::alloc(int _vertices_count, long long _edges_count)
{
    this->vertices_count = _vertices_count;
    this->edges_count = _edges_count;

    MemoryAPI::allocate_array(&reordered_vertex_ids, this->vertices_count);
    MemoryAPI::allocate_array(&outgoing_ptrs, this->vertices_count + 1);
    MemoryAPI::allocate_array(&outgoing_ids, this->edges_count);
    MemoryAPI::allocate_array(&incoming_degrees, this->vertices_count);
    MemoryAPI::allocate_array(&(this->vertex_values), this->vertices_count);
    
    #ifdef __USE_WEIGHTED_GRAPHS__
    MemoryAPI::allocate_array(&outgoing_weights, this->edges_count);
    #else
    outgoing_weights = NULL;
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::free()
{
    MemoryAPI::free_array(reordered_vertex_ids);
    MemoryAPI::free_array(outgoing_ptrs);
    MemoryAPI::free_array(outgoing_ids);
    MemoryAPI::free_array(incoming_degrees);
    MemoryAPI::free_array(this->vertex_values);

    #ifdef __USE_WEIGHTED_GRAPHS__
    MemoryAPI::free_array(outgoing_weights);
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::resize(int _vertices_count, long long _edges_count)
{
    this->free();
    this->alloc(_vertices_count, _edges_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::print()
{
    cout << "ExtendedCSRGraph format" << endl;
    
    cout << "VerticesState: " << this->vertices_state << endl;
    cout << "EdgesState: " << this->edges_state << endl;
    cout << "SupportedVectorLength: " << this->supported_vector_length << endl << endl;
    
    cout << "Vertices data: " << endl;
    for(int i = 0; i < this->vertices_count; i++)
        cout << this->vertex_values[i] << " ";
    cout << endl;

    cout << "Edges data: " << endl;
    for(int cur_vertex = 0; cur_vertex < this->vertices_count; cur_vertex++)
    {
        cout << "vertex " << cur_vertex << " connected to: ";
        for(long long edge_pos = outgoing_ptrs[cur_vertex]; edge_pos < outgoing_ptrs[cur_vertex + 1]; edge_pos++)
        {
            _TEdgeWeight weight = 0;
            #ifdef __USE_WEIGHTED_GRAPHS__
            weight = outgoing_weights[edge_pos];
            #endif
            cout << "(" << outgoing_ids[edge_pos] << "," << outgoing_weights[edge_pos] << ")" << " ";
        }
        cout << endl;
    }
    cout << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::save_to_graphviz_file(string _file_name, VisualisationMode _visualisation_mode)
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
    
        
    for(int i = 0; i < this->vertices_count; i++)
    {
        dot_output << i << " [label= \"id=" << i << ", value=" << this->vertex_values[i] << "\"] "<< endl;
        //dot_output << i << " [label=" << this->vertex_values[i] << "]"<< endl;
    }
    
    for(int cur_vertex = 0; cur_vertex < this->vertices_count; cur_vertex++)
    {
        int src_id = cur_vertex;
        for(long long edge_pos = outgoing_ptrs[cur_vertex]; edge_pos < outgoing_ptrs[cur_vertex + 1]; edge_pos++)
        {
            int dst_id = outgoing_ids[edge_pos];
            _TEdgeWeight weight = 0;
            #ifdef __USE_WEIGHTED_GRAPHS__
            weight = outgoing_weights[edge_pos];
            #endif
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

template <typename _TVertexValue, typename _TEdgeWeight>
bool ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::save_to_binary_file(string _file_name)
{
    FILE * graph_file = fopen(_file_name.c_str(), "wb");
    if(graph_file == NULL)
        return false;
    
    int vertices_count = this->vertices_count;
    long long edges_count = this->edges_count;
    fwrite(reinterpret_cast<const void*>(&vertices_count), sizeof(int), 1, graph_file);
    fwrite(reinterpret_cast<const void*>(&edges_count), sizeof(long long), 1, graph_file);
    
    fwrite(reinterpret_cast<const void*>(&(this->graph_type)), sizeof(GraphType), 1, graph_file);
    fwrite(reinterpret_cast<const void*>(&vertices_state), sizeof(VerticesState), 1, graph_file);
    fwrite(reinterpret_cast<const void*>(&edges_state), sizeof(EdgesState), 1, graph_file);
    fwrite(reinterpret_cast<const void*>(&supported_vector_length), sizeof(int), 1, graph_file);
    
    fwrite(reinterpret_cast<const char*>(this->vertex_values), sizeof(_TVertexValue), vertices_count, graph_file);
    fwrite(reinterpret_cast<const char*>(reordered_vertex_ids), sizeof(int), vertices_count, graph_file);
    fwrite(reinterpret_cast<const char*>(outgoing_ptrs), sizeof(long long), vertices_count + 1, graph_file);
    
    fwrite(reinterpret_cast<const char*>(outgoing_ids), sizeof(int), edges_count, graph_file);
    #ifdef __USE_WEIGHTED_GRAPHS__
    fwrite(reinterpret_cast<const char*>(outgoing_weights), sizeof(_TEdgeWeight), edges_count, graph_file);
    #endif
    
    fclose(graph_file);
    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
bool ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::load_from_binary_file(string _file_name)
{
    FILE * graph_file = fopen(_file_name.c_str(), "rb");
    if(graph_file == NULL)
        return false;
    
    fread(reinterpret_cast<void*>(&this->vertices_count), sizeof(int), 1, graph_file);
    fread(reinterpret_cast<void*>(&this->edges_count), sizeof(long long), 1, graph_file);
    
    this->resize(this->vertices_count, this->edges_count);
    
    fread(reinterpret_cast<void*>(&(this->graph_type)), sizeof(GraphType), 1, graph_file);
    
    if(this->graph_type != GraphTypeExtendedCSR)
        throw "ERROR: loaded incorrect type of graph into ExtendedCSRGraph container";
    
    fread(reinterpret_cast<void*>(&vertices_state), sizeof(VerticesState), 1, graph_file);
    fread(reinterpret_cast<void*>(&edges_state), sizeof(EdgesState), 1, graph_file);
    fread(reinterpret_cast<void*>(&supported_vector_length), sizeof(int), 1, graph_file);
    
    fread(reinterpret_cast<char*>(this->vertex_values), sizeof(_TVertexValue), this->vertices_count, graph_file);
    fread(reinterpret_cast<char*>(reordered_vertex_ids), sizeof(int), this->vertices_count, graph_file);
    fread(reinterpret_cast<char*>(outgoing_ptrs), sizeof(long long), (this->vertices_count + 1), graph_file);
    
    fread(reinterpret_cast<char*>(outgoing_ids), sizeof(int), this->edges_count, graph_file);
    #ifdef __USE_WEIGHTED_GRAPHS__
    fread(reinterpret_cast<char*>(outgoing_weights), sizeof(_TEdgeWeight), this->edges_count, graph_file);
    #endif
    
    calculate_incoming_degrees();
    #ifdef __USE_GPU__
    estimate_gpu_thresholds();
    #endif

    #ifdef __USE_NEC_SX_AURORA__
    double t1 = omp_get_wtime();
    estimate_nec_thresholds();
    last_vertices_ve.init_from_graph(this->outgoing_ptrs, this->outgoing_ids, this->outgoing_weights,
                                     nec_vector_core_threshold_vertex, this->vertices_count);
    double t2 = omp_get_wtime();
    cout << "NEC preprocess time: " << t2 - t1 << " sec" << endl;
    #endif
    
    fclose(graph_file);
    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


