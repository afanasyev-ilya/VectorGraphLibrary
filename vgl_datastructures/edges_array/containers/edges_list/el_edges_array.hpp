#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
EdgesArray_EL<_T>::EdgesArray_EL(VGL_Graph &_graph)
{
    this->total_array_size = _graph.get_edges_count() * 2;
    this->edges_count = _graph.get_edges_count();

    this->graph_ptr = &_graph;
    this->is_copy = false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
EdgesArray_EL<_T>::~EdgesArray_EL()
{
    if(!this->is_copy)
    {
        MemoryAPI::free_array(this->edges_data);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void EdgesArray_EL<_T>::set_all_constant(_T _const)
{
    MemoryAPI::set(this->edges_data, _const, this->total_array_size);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void EdgesArray_EL<_T>::set_all_random(_T _max_rand)
{
    // init CSR parts
    RandomGenerator rng_api;
    rng_api.generate_array_of_random_values<_T>(this->edges_data, this->total_array_size, _max_rand);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void EdgesArray_EL<_T>::set_equal_to_index()
{
    for(long long i = 0; i < this->total_array_size; i++)
    {
        this->edges_data[i] = i;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void EdgesArray_EL<_T>::print()
{
    cout << "Edges Array (Edges List)" << endl;
    for(long long i = 0; i < this->total_array_size; i++)
    {
        cout << this->edges_data[i] << " ";
    }
    cout << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void EdgesArray_EL<_T>::attach_pointer(_T *_outer_data)
{
    this->edges_data = _outer_data;

    outgoing_edges = this->edges_data;
    incoming_edges = &(this->edges_data[this->edges_count]);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
