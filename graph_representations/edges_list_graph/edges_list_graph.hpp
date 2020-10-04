#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
EdgesListGraph<_TVertexValue, _TEdgeWeight>::EdgesListGraph(int _vertices_count, long long _edges_count)
{
    this->graph_type = GraphTypeEdgesList;
    
    alloc(_vertices_count, _edges_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
EdgesListGraph<_TVertexValue, _TEdgeWeight>::~EdgesListGraph()
{
    free();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void EdgesListGraph<_TVertexValue, _TEdgeWeight>::alloc(int _vertices_count, long long _edges_count)
{
    this->vertices_count = _vertices_count;
    this->edges_count    = _edges_count;
    this->vertex_values  = new _TVertexValue[this->vertices_count];
    src_ids              = new int[this->edges_count];
    dst_ids              = new int[this->edges_count];
    weights              = new _TEdgeWeight[this->edges_count];
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void EdgesListGraph<_TVertexValue, _TEdgeWeight>::free()
{
    if(this->vertex_values != NULL)
        delete []this->vertex_values;
    if(src_ids != NULL)
        delete []src_ids;
    if(dst_ids != NULL)
        delete []dst_ids;
    if(weights != NULL)
        delete []weights;
    
    this->vertex_values = NULL;
    src_ids = NULL;
    dst_ids = NULL;
    weights = NULL;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void EdgesListGraph<_TVertexValue, _TEdgeWeight>::resize(int _vertices_count, long long _edges_count)
{
    this->free();
    this->alloc(_vertices_count, _edges_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void EdgesListGraph<_TVertexValue, _TEdgeWeight>::save_to_graphviz_file(string _file_name, VisualisationMode _visualisation_mode)
{
    ofstream dot_output(_file_name.c_str());
    
    string connection;
    dot_output << "digraph G {" << endl;
    connection = " -> ";
    
    for(int i = 0; i < this->vertices_count; i++)
    {
        dot_output << i << " [label= \"id=" << i << ", value=" << this->vertex_values[i] << "\"] "<< endl;
        //dot_output << i << " [label=" << this->vertex_values[i] << "]"<< endl;
    }
    
    for(long long i = 0; i < this->edges_count; i++)
    {
        int src_id = src_ids[i];
        int dst_id = dst_ids[i];
        _TEdgeWeight weight = weights[i];
        dot_output << src_id << connection << dst_id << " [label = \" " << weight << " \"];" << endl;
    }
    
    dot_output << "}";
    dot_output.close();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
bool EdgesListGraph<_TVertexValue, _TEdgeWeight>::save_to_binary_file(string _file_name)
{
    FILE * graph_file = fopen(_file_name.c_str(), "wb");
    if(graph_file == NULL)
        return false;
    
    int vertices_count = this->vertices_count;
    long long edges_count = this->edges_count;
    fwrite(reinterpret_cast<const void*>(&vertices_count), sizeof(int), 1, graph_file);
    fwrite(reinterpret_cast<const void*>(&edges_count), sizeof(long long), 1, graph_file);

    fwrite(reinterpret_cast<const void*>(src_ids), sizeof(int), this->edges_count, graph_file);
    fwrite(reinterpret_cast<const void*>(dst_ids), sizeof(int), this->edges_count, graph_file);
    fwrite(reinterpret_cast<const void*>(weights), sizeof(_TEdgeWeight), this->edges_count, graph_file);

    fclose(graph_file);
    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
bool EdgesListGraph<_TVertexValue, _TEdgeWeight>::load_from_binary_file(string _file_name)
{
    FILE * graph_file = fopen(_file_name.c_str(), "rb");
    if(graph_file == NULL)
        return false;
    
    fread(reinterpret_cast<void*>(&this->vertices_count), sizeof(int), 1, graph_file);
    fread(reinterpret_cast<void*>(&this->edges_count), sizeof(long long), 1, graph_file);
    
    fread(reinterpret_cast<void*>(src_ids), sizeof(int), this->edges_count, graph_file);
    fread(reinterpret_cast<void*>(dst_ids), sizeof(int), this->edges_count, graph_file);
    fread(reinterpret_cast<void*>(weights), sizeof(_TEdgeWeight), this->edges_count, graph_file);
    
    fclose(graph_file);
    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void EdgesListGraph<_TVertexValue, _TEdgeWeight>::transpose()
{
    int *tmp_ptr = src_ids;
    src_ids = dst_ids;
    dst_ids = tmp_ptr;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void EdgesListGraph<_TVertexValue, _TEdgeWeight>::renumber_vertices(int *_conversion_array, int *_work_buffer)
{
    double t1 = omp_get_wtime();

    // TODO reorder vertex values
    // TODO save conversion arrays

    bool work_buffer_was_allocated = false;
    if(_work_buffer == NULL)
    {
        work_buffer_was_allocated = true;
        MemoryAPI::allocate_array(&_work_buffer, this->edges_count);
    }

    #pragma _NEC ivdep
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma _NEC gather_reorder
    #pragma omp parallel for
    for(long long edge_pos = 0; edge_pos < this->edges_count; edge_pos++)
    {
        _work_buffer[edge_pos] = _conversion_array[src_ids[edge_pos]];
    }

    #pragma _NEC ivdep
    #pragma omp parallel for
    for(long long edge_pos = 0; edge_pos < this->edges_count; edge_pos++)
    {
        src_ids[edge_pos] = _work_buffer[edge_pos];
    }

    #pragma _NEC ivdep
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma _NEC gather_reorder
    #pragma omp parallel for
    for(long long edge_pos = 0; edge_pos < this->edges_count; edge_pos++)
    {
        _work_buffer[edge_pos] = _conversion_array[dst_ids[edge_pos]];
    }

    #pragma _NEC ivdep
    #pragma omp parallel for
    for(long long edge_pos = 0; edge_pos < this->edges_count; edge_pos++)
    {
        dst_ids[edge_pos] = _work_buffer[edge_pos];
    }

    if(work_buffer_was_allocated)
    {
        MemoryAPI::free_array(_work_buffer);
    }

    double t2 = omp_get_wtime();
    cout << "edges list graph reorder (to optimized) time: " << t2 - t1 << " sec" << endl;
    cout << "BW: " << this->edges_count*sizeof(int)*(2*2 + 3*2)/((t2 - t1)*1e9) << " GB/s" << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
