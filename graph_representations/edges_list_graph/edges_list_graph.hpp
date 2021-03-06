#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

EdgesListGraph::EdgesListGraph(int _vertices_count, long long _edges_count)
{
    this->graph_type = EDGES_LIST_GRAPH;
    this->supported_direction = USE_BOTH;

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
    MemoryAPI::allocate_array(&src_ids, this->edges_count);
    MemoryAPI::allocate_array(&dst_ids, this->edges_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void EdgesListGraph::free()
{
    MemoryAPI::free_array(src_ids);
    MemoryAPI::free_array(dst_ids);

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

    int vertices_count = this->vertices_count;
    long long edges_count = this->edges_count;
    fwrite(reinterpret_cast<void*>(&this->graph_type), sizeof(GraphType), 1, graph_file);
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

    fread(reinterpret_cast<void*>(&this->graph_type), sizeof(GraphType), 1, graph_file);
    fread(reinterpret_cast<void*>(&this->vertices_count), sizeof(int), 1, graph_file);
    fread(reinterpret_cast<void*>(&this->edges_count), sizeof(long long), 1, graph_file);

    resize(this->vertices_count, this->edges_count);
    
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
    Timer tm;
    tm.start();

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

    tm.end();
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_time_and_bandwidth_stats("EdgesList graph reorder (to optimized)", this->edges_count, sizeof(int)*(2*2 + 3*2));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void EdgesListGraph::operator = (const EdgesListGraph &_copy_graph)
{
    this->graph_type = _copy_graph.graph_type;
    this->vertices_count = _copy_graph.vertices_count;
    this->edges_count = _copy_graph.edges_count;
    this->graph_on_device = _copy_graph.graph_on_device;

    this->resize(this->vertices_count, this->edges_count);

    MemoryAPI::copy(this->src_ids, _copy_graph.src_ids, this->edges_count);
    MemoryAPI::copy(this->dst_ids, _copy_graph.dst_ids, this->edges_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

EdgesListGraph::EdgesListGraph(const EdgesListGraph &_copy_graph)
{
    this->graph_type = _copy_graph.graph_type;
    this->vertices_count = _copy_graph.vertices_count;
    this->edges_count = _copy_graph.edges_count;
    this->graph_on_device = _copy_graph.graph_on_device;

    this->alloc(this->vertices_count, this->edges_count);

    MemoryAPI::copy(this->src_ids, _copy_graph.src_ids, this->edges_count);
    MemoryAPI::copy(this->dst_ids, _copy_graph.dst_ids, this->edges_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

