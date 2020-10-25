#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

VectCSRGraph::VectCSRGraph(int _vertices_count, long long _edges_count)
{
    this->graph_type = VECT_CSR_GRAPH;

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

    outgoing_graph = new UndirectedCSRGraph(this->vertices_count, this->edges_count );
    incoming_graph = new UndirectedCSRGraph(this->vertices_count, this->edges_count );

    MemoryAPI::allocate_array(&vertices_reorder_buffer, this->vertices_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VectCSRGraph::free()
{
    delete outgoing_graph;
    delete incoming_graph;

    if(vertices_reorder_buffer != NULL)
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
    outgoing_graph->save_to_graphviz_file(_file_name, _vertex_data, VISUALISE_AS_DIRECTED);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

UndirectedCSRGraph *VectCSRGraph::get_direction_graph_ptr(TraversalDirection _direction)
{
    if(_direction == SCATTER)
    {
        return get_outgoing_graph_ptr();
    }
    else if(_direction == GATHER)
    {
        return get_incoming_graph_ptr();
    }
    else
    {
        throw "Error in UndirectedCSRGraph::get_direction_graph_ptr, incorrect _direction type";
        return NULL;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int VectCSRGraph::select_random_vertex(TraversalDirection _direction)
{
    int attempt_num = 0;
    while(attempt_num < ATTEMPTS_THRESHOLD)
    {
        int outgoing_vertex_id = outgoing_graph->select_random_vertex();
        int incoming_vertex_id = this->reorder(outgoing_vertex_id, SCATTER, GATHER);
        if(incoming_graph->get_connections_count(incoming_vertex_id) > 0)
        {
            int original_vertex_id = this->reorder(outgoing_vertex_id, SCATTER, ORIGINAL);
            return original_vertex_id;
        }

        attempt_num++;
    }

    throw "Error in VectCSRGraph::select_random_vertex: can not select non-zero degree vertex in ATTEMPTS_THRESHOLD attempts";

    return -1;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

