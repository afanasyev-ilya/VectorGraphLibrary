/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

UndirectedCSRGraph *VectCSRGraph::get_outgoing_graph_ptr()
{
    if(outgoing_is_stored())
    {
        return outgoing_graph;
    }
    throw "Error in VectCSRGraph::get_outgoing_graph_ptr : outgoing graph is not stored";
    return NULL;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

UndirectedCSRGraph *VectCSRGraph::get_incoming_graph_ptr()
{
    if(incoming_is_stored())
    {
        return incoming_graph;
    }
    throw "Error in VectCSRGraph::get_incoming_graph_ptr : incoming graph is not stored";
    return NULL;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

UndirectedCSRGraph *VectCSRGraph::get_direction_graph_ptr(TraversalDirection _direction)
{
    if(_direction == SCATTER)
        return this->get_outgoing_graph_ptr();
    else if(_direction == GATHER)
        return this->get_incoming_graph_ptr();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline long long VectCSRGraph::get_edges_count_in_outgoing_ve()
{
    if(outgoing_is_stored())
        return outgoing_graph->get_edges_count_in_ve();
    else
        return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline long long VectCSRGraph::get_edges_count_in_incoming_ve()
{
    if(incoming_is_stored())
        return incoming_graph->get_edges_count_in_ve();
    else
        return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline long long VectCSRGraph::get_edges_count_in_outgoing_csr()
{
    if(outgoing_is_stored())
        return this->edges_count;
    else
        return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline long long VectCSRGraph::get_edges_count_in_incoming_csr()
{
    if(incoming_is_stored())
        return this->edges_count;
    else
        return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
