#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
VerticesArrayNec<_T>::VerticesArrayNec(VectCSRGraph &_graph, DataDirection _direction)
{
    direction = _direction;
    vertices_count = _graph.get_vertices_count();
    MemoryAPI::allocate_array(&vertices_data, vertices_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
VerticesArrayNec<_T>::VerticesArrayNec(const VerticesArrayNec<_T> &_copy_obj)
{
    this->graph_ptr = _copy_obj.graph_ptr;
    this->vertices_count = _copy_obj.vertices_count;
    this->direction = _copy_obj.direction;
    MemoryAPI::allocate_array(&this->vertices_data, vertices_count);
    MemoryAPI::copy(this->vertices_data, _copy_obj.vertices_data, vertices_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
VerticesArrayNec<_T>::~VerticesArrayNec()
{
    MemoryAPI::free_array(vertices_data);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void VerticesArrayNec<_T>::set_all_constant(_T _const)
{
    MemoryAPI::set(vertices_data, _const, vertices_count);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void VerticesArrayNec<_T>::set_all_random(_T _max_rand)
{
    // init CSR parts
    RandomGenerationAPI rng_api;
    rng_api.generate_array_of_random_values<_T>(vertices_data, vertoces_count, _max_rand);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void VerticesArrayNec<_T>::print()
{
    for(int i = 0; i < vertices_count; i++)
    {
        cout << vertices_data[i] << " ";
    }
    cout << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
