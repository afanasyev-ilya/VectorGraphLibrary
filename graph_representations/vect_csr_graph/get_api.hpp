/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

VectorCSRGraph *VectCSRGraph::get_outgoing_data()
{
    if(outgoing_is_stored())
    {
        return outgoing_graph;
    }
    throw "Error in VectCSRGraph::get_outgoing_data : outgoing graph is not stored";
    return NULL;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

VectorCSRGraph *VectCSRGraph::get_incoming_data()
{
    if(incoming_is_stored())
    {
        return incoming_graph;
    }
    throw "Error in VectCSRGraph::get_incoming_data : incoming graph is not stored";
    return NULL;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

VectorCSRGraph *VectCSRGraph::get_direction_graph_ptr(TraversalDirection _direction)
{
    if(_direction == SCATTER)
        return this->get_outgoing_data();
    else if(_direction == GATHER)
        return this->get_incoming_data();
    throw "Error in VectCSRGraph::get_direction_graph_ptr : incorrect _direction value";
    return NULL;
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

int VectCSRGraph::get_edge_dst(int _src_id, int _local_edge_pos, TraversalDirection _direction)
{
    if(_direction == SCATTER)
        return this->outgoing_graph->get_edge_dst(_src_id, _local_edge_pos);
    else if(_direction == GATHER)
        return this->incoming_graph->get_edge_dst(_src_id, _local_edge_pos);
    return -1;
}

int VectCSRGraph::get_incoming_edge_dst(int _src_id, int _local_edge_pos)
{
    return this->incoming_graph->get_edge_dst(_src_id, _local_edge_pos);
}

int VectCSRGraph::get_outgoing_edge_dst(int _src_id, int _local_edge_pos)
{
    return this->outgoing_graph->get_edge_dst(_src_id, _local_edge_pos);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int VectCSRGraph::get_connections_count(int _src_id, TraversalDirection _direction)
{
    if(_direction == SCATTER)
        return this->outgoing_graph->get_connections_count(_src_id);
    else if(_direction == GATHER)
        return this->incoming_graph->get_connections_count(_src_id);
    return -1;
}

int VectCSRGraph::get_incoming_connections_count(int _src_id)
{
    return this->incoming_graph->get_connections_count(_src_id);
}

int VectCSRGraph::get_outgoing_connections_count(int _src_id)
{
    return this->outgoing_graph->get_connections_count(_src_id);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
