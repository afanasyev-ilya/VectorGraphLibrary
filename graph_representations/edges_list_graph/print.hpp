#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void EdgesListGraph::print()
{
    cout << endl;
    cout << "Graph in edges list format" << endl;
    cout << "|V|=" << this->vertices_count << endl;
    cout << "|E|=" << this->edges_count << endl;
    for(long long int i = 0; i < this->edges_count; i++)
        cout << i << ") " << src_ids[i] << " " << dst_ids[i] << endl;
    cout << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void EdgesListGraph::print_in_csr_format()
{
    this->preprocess_into_csr_based();

    cout << endl;
    cout << "Graph in edges list format" << endl;
    cout << "|V|=" << this->vertices_count << endl;
    cout << "|E|=" << this->edges_count << endl;
    int current_src_id = -1;
    for(long long int i = 0; i < this->edges_count; i++)
    {
        if(src_ids[i] != current_src_id)
        {
            if(current_src_id != -1)
                cout << endl;
            cout << "vertex " << src_ids[i] << " is connected to: ";
            current_src_id = src_ids[i];
        }
        cout << dst_ids[i] << ", ";
    }
    cout << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void EdgesListGraph::print_in_csr_format(EdgesArray_EL<_T> &_weights)
{
    this->preprocess_into_csr_based();

    cout << endl;
    cout << "Graph in edges list format" << endl;
    cout << "|V|=" << this->vertices_count << endl;
    cout << "|E|=" << this->edges_count << endl;
    int current_src_id = -1;
    for(long long int i = 0; i < this->edges_count; i++)
    {
        if(src_ids[i] != current_src_id)
        {
            if(current_src_id != -1)
                cout << endl;
            cout << "vertex " << src_ids[i] << " is connected to: ";
            current_src_id = src_ids[i];
        }
        cout << "(" << dst_ids[i] << ", " << _weights[i] << ") ";
    }
    cout << endl << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void EdgesListGraph::print_size()
{
    cout << "Wall size (EdgesListGraph): " << get_size()/1e9 << " GB" << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

size_t EdgesListGraph::get_size()
{
    size_t graph_size = this->edges_count*(sizeof(src_ids[0]) + sizeof(dst_ids[0]));
    return graph_size;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
