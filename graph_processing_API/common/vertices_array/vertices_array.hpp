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

    is_copy = false;

    #ifdef __USE_NEC_SX_AURORA__
    #pragma omp parallel
    {};
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
VerticesArray<_T>::VerticesArray(ShardedCSRGraph &_graph, TraversalDirection _direction)
{
    object_type = VERTICES_ARRAY;
    graph_ptr = &_graph;

    this->direction = _direction;
    this->vertices_count = _graph.get_vertices_count();
    MemoryAPI::allocate_array(&this->vertices_data, this->vertices_count);

    is_copy = false;

    #ifdef __USE_NEC_SX_AURORA__
    #pragma omp parallel
    {};
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
VerticesArray<_T>::VerticesArray(EdgesListGraph &_graph, TraversalDirection _direction)
{
    object_type = VERTICES_ARRAY;
    graph_ptr = &_graph;

    this->direction = _direction;
    this->vertices_count = _graph.get_vertices_count();
    MemoryAPI::allocate_array(&this->vertices_data, this->vertices_count);

    is_copy = false;

    #ifdef __USE_NEC_SX_AURORA__
    #pragma omp parallel
    {};
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
VerticesArray<_T>::VerticesArray(const VerticesArray<_T> &_copy_obj)
{
    this->object_type = _copy_obj.object_type;
    this->vertices_count = _copy_obj.vertices_count;
    this->direction = _copy_obj.direction;
    this->vertices_data = _copy_obj.vertices_data;
    this->is_copy = true;

    #ifdef __USE_NEC_SX_AURORA__
    #pragma omp parallel
    {};
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
VerticesArray<_T>::~VerticesArray()
{
    if(!is_copy)
    {
        MemoryAPI::free_array(this->vertices_data);
        this->vertices_data = NULL;
    }
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
    RandomGenerator rng_api;
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

template class VerticesArray<int>;
template class VerticesArray<float>;
template class VerticesArray<double>;
template class VerticesArray<long long>;
template class VerticesArray<bool>;
template class VerticesArray<char>;
template class VerticesArray<short>;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
