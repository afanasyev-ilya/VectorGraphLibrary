#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VectCSRGraph::print()
{
    outgoing_graph->print();
    incoming_graph->print();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void VectCSRGraph::print_with_weights(EdgesArray<_T> &_weights)
{
    outgoing_graph->print_with_weights(_weights, SCATTER);
    incoming_graph->print_with_weights(_weights, GATHER);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VectCSRGraph::print_size()
{
    cout << endl;
    cout << " --------------------- Graph size --------------------- " << endl;
    cout << "Outgoing graph size: " << endl;
    outgoing_graph->print_size();
    cout << "Incoming graph size: " << endl;
    incoming_graph->print_size();
    cout << "Wall size (VectCSRGraph): " << get_size()/1e9 << " GB" << endl;
    cout << " ------------------------------------------------------ " << endl;
    cout << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VectCSRGraph::print_stats()
{
    cout << endl;
    cout << " --------------------- Graph stats --------------------- " << endl;
    outgoing_graph->print_stats();
    cout << " ------------------------------------------------------- " << endl;
    cout << endl;

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

size_t VectCSRGraph::get_size()
{
    size_t size = outgoing_graph->get_size() + incoming_graph->get_size();
    size += sizeof(vertices_reorder_buffer[0])*this->vertices_count;
    return size;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VectCSRGraph::print_vertex_information(TraversalDirection _direction, int _src_id, int _num_edges)
{
    get_direction_graph_ptr(_direction)->print_vertex_information(_src_id, _num_edges);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


