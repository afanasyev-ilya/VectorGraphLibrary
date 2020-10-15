#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VectCSRGraph::print()
{
    outgoing_graph->print();
    incoming_graph->print();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void VectCSRGraph::print_with_weights(EdgesArrayNEC<_T> &_weights)
{
    outgoing_graph->print_with_weights(_weights, SCATTER);
    incoming_graph->print_with_weights(_weights, GATHER);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VectCSRGraph::print_size()
{
    cout << "Outgoing graph size: " << outgoing_graph->get_size()/1e9 << " GB" << endl;
    cout << "Incoming graph size: " << incoming_graph->get_size()/1e9 << " GB" << endl;
    cout << "Wall size: " << get_size()/1e9 << " GB" << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

size_t VectCSRGraph::get_size()
{
    size_t size = outgoing_graph->get_size() + incoming_graph->get_size();
    size += sizeof(vertices_reorder_buffer[0])*this->vertices_count;
    size += sizeof(edges_reorder_indexes[0])*this->edges_count;
    return size;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
