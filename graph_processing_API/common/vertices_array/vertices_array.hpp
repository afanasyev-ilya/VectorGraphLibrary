#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
VerticesArray<_T>::VerticesArray(VectCSRGraph &_graph, TraversalDirection _direction)
{
    object_type = VERTICES_ARRAY;
    graph_ptr = &_graph;

    this->direction = _direction;
    this->vertices_count = _graph.get_vertices_count();
    MemoryAPI::allocate_array(&this->vertices_data, this->vertices_count);
    #pragma omp parallel
    {};
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
VerticesArray<_T>::VerticesArray(const VerticesArray<_T> &_copy_obj)
{
    this->object_type = _copy_obj.object_type;
    this->graph_ptr = _copy_obj.graph_ptr;
    this->vertices_count = _copy_obj.vertices_count;
    this->direction = _copy_obj.direction;
    MemoryAPI::allocate_array(&this->vertices_data, this->vertices_count);
    MemoryAPI::copy(this->vertices_data, _copy_obj.vertices_data, this->vertices_count);
    #pragma omp parallel
    {};
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
VerticesArray<_T>::~VerticesArray()
{
    MemoryAPI::free_array(this->vertices_data);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void VerticesArray<_T>::set_all_constant(_T _const)
{
    MemoryAPI::set(this->vertices_data, _const, this->vertices_count);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void VerticesArray<_T>::set_all_random(_T _max_rand)
{
    // init CSR parts
    RandomGenerationAPI rng_api;
    rng_api.generate_array_of_random_values<_T>(this->vertices_data, this->vertices_count, _max_rand);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void VerticesArray<_T>::print()
{
    for(int i = 0; i < this->vertices_count; i++)
    {
        cout << this->vertices_data[i] << " ";
    }
    cout << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void VerticesArray<_T>::print(string _name)
{
    cout << _name << ": ";
    print();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
