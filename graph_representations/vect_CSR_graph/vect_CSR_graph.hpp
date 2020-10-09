#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

VectCSRGraph::VectCSRGraph(int _vertices_count, long long _edges_count)
{
    this->vertices_count = _vertices_count;
    this->edges_count = _edges_count;
    outgoing_graph = new ExtendedCSRGraph(_vertices_count, _edges_count);
    incoming_graph = new ExtendedCSRGraph(_vertices_count, _edges_count);

    edges_reorder_buffer = NULL;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

VectCSRGraph::~VectCSRGraph()
{
    delete outgoing_graph;
    delete incoming_graph;

    if(edges_reorder_buffer != NULL)
        MemoryAPI::free_array(edges_reorder_buffer);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ExtendedCSRGraph *VectCSRGraph::get_direction_graph_ptr(TraversalDirection _direction)
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
        throw "Error in ExtendedCSRGraph::get_direction_graph_ptr, incorrect _direction type";
        return NULL;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
