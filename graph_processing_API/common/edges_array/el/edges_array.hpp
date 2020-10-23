#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
EdgesArray_EL<_T>::EdgesArray_EL(EdgesListGraph &_graph)
{
    this->total_array_size = _graph.get_edges_count();

    MemoryAPI::allocate_array(&this->edges_data, this->total_array_size);

    this->graph_ptr = &_graph;
    this->is_copy = false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
EdgesArray_EL<_T>::EdgesArray_EL(const EdgesArray_EL<_T> &_copy_obj)
{
    this->graph_ptr = _copy_obj.graph_ptr;
    this->edges_data = _copy_obj.edges_data;
    this->total_array_size = _copy_obj.total_array_size;
    this->is_copy = true;
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

template class EdgesArray_EL<int>;
template class EdgesArray_EL<float>;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
