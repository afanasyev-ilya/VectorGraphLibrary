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
    outgoing_data->import(_edges_container);
    incoming_data->import(_edges_container);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VGL_Graph::print()
{
    cout << " ------------ VGL GRAPH ------------ " << endl;
    outgoing_data->print();
    incoming_data->print();
    cout << " ------------ VGL GRAPH ------------ " << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

