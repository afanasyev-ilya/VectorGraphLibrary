/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

VGL_Graph::VGL_Graph(GraphType _container_type)
{
    graph_type = VGL_GRAPH;
    if(_container_type == VECTOR_CSR_GRAPH)
    {
        outgoing_data = new VectorCSRGraph();
        incoming_data = new VectorCSRGraph();
    }
    else if(_container_type == EDGES_LIST_GRAPH)
    {
        outgoing_data = new EdgesListGraph();
        incoming_data = new EdgesListGraph();
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VGL_Graph::import(EdgesContainer &_edges_container)
{
    this->vertices_count = _edges_container.get_vertices_count();
    this->edges_count = _edges_container.get_edges_count();
    outgoing_data->import(_edges_container);
    incoming_data->import(_edges_container);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VGL_Graph::print()
{
    outgoing_data->print();
    incoming_data->print();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VGL_Graph::print_size()
{
    outgoing_data->print_size();
    incoming_data->print_size();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

UndirectedGraph *VGL_Graph::get_direction_data(TraversalDirection _direction)
{
    if(_direction == SCATTER)
        return outgoing_data;
    else
        return incoming_data;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


