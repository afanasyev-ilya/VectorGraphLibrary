#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VectCSRGraph::print()
{
    if(outgoing_is_stored())
        outgoing_graph->print();
    if(incoming_is_stored())
        incoming_graph->print();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void VectCSRGraph::print_with_weights(EdgesArray<_T> &_weights)
{
    if(outgoing_is_stored())
        outgoing_graph->print_with_weights(_weights, SCATTER);
    if(incoming_is_stored())
        incoming_graph->print_with_weights(_weights, GATHER);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VectCSRGraph::print_size()
{
    cout << endl;
    cout << " --------------------- Graph size --------------------- " << endl;
    if(outgoing_is_stored())
    {
        cout << "Outgoing graph size: " << endl;
        outgoing_graph->print_size();
    }
    if(incoming_is_stored())
    {
        cout << "Incoming graph size: " << endl;
        incoming_graph->print_size();
    }
    cout << "Wall size (VectCSRGraph): " << get_size()/1e9 << " GB" << endl;
    cout << " ------------------------------------------------------ " << endl;
    cout << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VectCSRGraph::print_stats()
{
    cout << endl;
    cout << " --------------------- Graph stats --------------------- " << endl;
    if(outgoing_is_stored())
    {
        outgoing_graph->print_stats();
    }
    if(incoming_is_stored())
    {
        incoming_graph->print_stats();
    }
    cout << " ------------------------------------------------------- " << endl;
    cout << endl;

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

size_t VectCSRGraph::get_size()
{
    size_t outgoing_size = 0;
    size_t incoming_size = 0;
    if(outgoing_is_stored())
        outgoing_size = outgoing_graph->get_size();
    if(incoming_is_stored())
        incoming_size = incoming_graph->get_size() + sizeof(vertices_reorder_buffer[0])*this->vertices_count;

    size_t size = outgoing_size + incoming_size;
    return size;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VectCSRGraph::print_vertex_information(TraversalDirection _direction, int _src_id, int _num_edges)
{
    get_direction_graph_ptr(_direction)->print_vertex_information(_src_id, _num_edges);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


