#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
EdgesArray_CSR<_T>::EdgesArray_CSR(VGL_Graph &_graph)
{
    this->graph_ptr = &_graph;
    this->edges_count = _graph.get_edges_count();
    this->total_array_size = _graph.get_edges_count() * _graph.get_number_of_directions();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
EdgesArray_CSR<_T>::~EdgesArray_CSR()
{

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void EdgesArray_CSR<_T>::set_all_constant(_T _const)
{
    MemoryAPI::set(this->edges_data, _const, this->total_array_size);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void EdgesArray_CSR<_T>::set_all_random(_T _max_rand)
{
    RandomGenerator rng_api;
    rng_api.generate_array_of_random_values<_T>(this->outgoing_edges, this->edges_count, _max_rand);
    if(this->graph_ptr->get_number_of_directions() == BOTH_DIRECTIONS)
    {
        this->graph_ptr->copy_outgoing_to_incoming_edges(this->outgoing_edges, this->incoming_edges);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void EdgesArray_CSR<_T>::print()
{
    cout << "Edges Array (CSR)" << endl;
    cout << "outgoing: ";
    for(long long i = 0; i < this->edges_count; i++)
    {
        cout << this->outgoing_edges[i] << " ";
    }
    cout << endl << endl;
    if(this->graph_ptr->get_number_of_directions() == BOTH_DIRECTIONS)
    {
        cout << "incoming: ";
        for(long long i = 0; i < this->edges_count; i++)
        {
            cout << this->incoming_edges[i] << " ";
        }
        cout << endl << endl;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void EdgesArray_CSR<_T>::attach_pointer(_T *_outer_data)
{
    this->edges_data = _outer_data;

    this->outgoing_edges = this->edges_data;
    this->incoming_edges = &(this->edges_data[this->edges_count]);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


