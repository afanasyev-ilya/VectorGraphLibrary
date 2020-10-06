#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

EdgesListGraph::EdgesListGraph(int _vertices_count, long long _edges_count)
{
    this->graph_type = GraphTypeEdgesList;
    
    alloc(_vertices_count, _edges_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

EdgesListGraph::~EdgesListGraph()
{
    free();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void EdgesListGraph::alloc(int _vertices_count, long long _edges_count)
{
    this->vertices_count = _vertices_count;
    this->edges_count    = _edges_count;
    src_ids              = new int[this->edges_count]; // TODO correct alloc
    dst_ids              = new int[this->edges_count];
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void EdgesListGraph::free()
{
    if(src_ids != NULL)
        delete []src_ids;
    if(dst_ids != NULL)
        delete []dst_ids;

    src_ids = NULL;
    dst_ids = NULL;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void EdgesListGraph::resize(int _vertices_count, long long _edges_count)
{
    this->free();
    this->alloc(_vertices_count, _edges_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void EdgesListGraph::save_to_graphviz_file(string _file_name, VisualisationMode _visualisation_mode)
{
    ofstream dot_output(_file_name.c_str());
    
    string connection;
    dot_output << "digraph G {" << endl;
    connection = " -> ";
    
    for(long long i = 0; i < this->edges_count; i++)
    {
        int src_id = src_ids[i];
        int dst_id = dst_ids[i];
        dot_output << src_id << connection << dst_id << " [label = \" " << " TODO weight " << " \"];" << endl;
    }
    
    dot_output << "}";
    dot_output.close();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool EdgesListGraph::save_to_binary_file(string _file_name)
{
    FILE * graph_file = fopen(_file_name.c_str(), "wb");
    if(graph_file == NULL)
        return false;

    // TODO graph type
    int vertices_count = this->vertices_count;
    long long edges_count = this->edges_count;
    fwrite(reinterpret_cast<const void*>(&vertices_count), sizeof(int), 1, graph_file);
    fwrite(reinterpret_cast<const void*>(&edges_count), sizeof(long long), 1, graph_file);

    fwrite(reinterpret_cast<const void*>(src_ids), sizeof(int), this->edges_count, graph_file);
    fwrite(reinterpret_cast<const void*>(dst_ids), sizeof(int), this->edges_count, graph_file);

    fclose(graph_file);
    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool EdgesListGraph::load_from_binary_file(string _file_name)
{
    FILE * graph_file = fopen(_file_name.c_str(), "rb");
    if(graph_file == NULL)
        return false;
    
    fread(reinterpret_cast<void*>(&this->vertices_count), sizeof(int), 1, graph_file);
    fread(reinterpret_cast<void*>(&this->edges_count), sizeof(long long), 1, graph_file);
    
    fread(reinterpret_cast<void*>(src_ids), sizeof(int), this->edges_count, graph_file);
    fread(reinterpret_cast<void*>(dst_ids), sizeof(int), this->edges_count, graph_file);
    
    fclose(graph_file);
    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void EdgesListGraph::transpose()
{
    int *tmp_ptr = src_ids;
    src_ids = dst_ids;
    dst_ids = tmp_ptr;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void EdgesListGraph::renumber_vertices(int *_conversion_array, int *_work_buffer)
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
