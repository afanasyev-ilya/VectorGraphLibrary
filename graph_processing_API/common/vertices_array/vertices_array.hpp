#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
VerticesArray<_T>::VerticesArray(VectCSRGraph &_graph, TraversalDirection _direction, CachedMode _cached_mode)
{
    object_type = VERTICES_ARRAY;
    graph_ptr = &_graph;

    this->direction = _direction;
    this->vertices_count = _graph.get_vertices_count();
    MemoryAPI::allocate_array(&this->vertices_data, this->vertices_count);

    this->cached_mode = _cached_mode;
    allocate_cached_array();

    is_copy = false;

    #ifdef __USE_NEC_SX_AURORA__
    #pragma omp parallel
    {};
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
VerticesArray<_T>::VerticesArray(ShardedCSRGraph &_graph, TraversalDirection _direction, CachedMode _cached_mode)
{
    object_type = VERTICES_ARRAY;
    graph_ptr = &_graph;

    this->direction = _direction;
    this->vertices_count = _graph.get_vertices_count();
    MemoryAPI::allocate_array(&this->vertices_data, this->vertices_count);

    is_copy = false;

    this->cached_mode = _cached_mode;
    allocate_cached_array();

    #ifdef __USE_NEC_SX_AURORA__
    #pragma omp parallel
    {};
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
VerticesArray<_T>::VerticesArray(EdgesListGraph &_graph, TraversalDirection _direction, CachedMode _cached_mode)
{
    object_type = VERTICES_ARRAY;
    graph_ptr = &_graph;

    this->direction = _direction;
    this->vertices_count = _graph.get_vertices_count();
    MemoryAPI::allocate_array(&this->vertices_data, this->vertices_count);

    is_copy = false;

    this->cached_mode = _cached_mode;
    allocate_cached_array();

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

    this->cached_mode = _copy_obj.cached_mode;
    this->cached_data = _copy_obj.cached_data;

    #ifdef __USE_NEC_SX_AURORA__
    #pragma omp parallel
    {};
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void VerticesArray<_T>::allocate_cached_array()
{
    if(this->cached_mode == USE_CACHED_MODE)
    {
        int threads_count = omp_get_max_threads();

        MemoryAPI::allocate_array(&this->cached_data, threads_count * CACHED_VERTICES * CACHE_STEP);

        #ifdef __USE_NEC_SX_AURORA__
        #pragma retain(this->cached_data)
        #endif
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void VerticesArray<_T>::free_cached_array()
{
    if(this->cached_mode == USE_CACHED_MODE)
    {
        MemoryAPI::free_array(this->cached_data);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
VerticesArray<_T>::~VerticesArray()
{
    if(!is_copy)
    {
        MemoryAPI::free_array(this->vertices_data);
        free_cached_array();
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

template <typename _T>
_T VerticesArray<_T>::cached_load(int _idx, _T *_private_data)
{
    _T result = 0;
    if(_idx < CACHED_VERTICES)
        result = _private_data[_idx * CACHE_STEP];
    else
        result = vertices_data[_idx];

    return result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void VerticesArray<_T>::prefetch_data_into_cache()
{
    if(omp_in_parallel())
    {
        _T *private_data = this->get_private_data_pointer();

        #ifdef __USE_NEC_SX_AURORA__
        #pragma _NEC vector
        #endif
        for(int i = 0; i < CACHED_VERTICES; i++)
            private_data[i * CACHE_STEP] = vertices_data[i];
    }
    else
    {
        #pragma omp parallel
        {
            this->prefetch_data_into_cache();
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <class _T>
_T* VerticesArray<_T>::get_private_data_pointer()
{
    if(!omp_in_parallel())
    {
        throw "Error in VerticesArray<_T>::get_private_data_pointer : should be used only in parallel region";
        return NULL;
    }

    int thread_id = omp_get_thread_num();
    _T *private_data = &cached_data[thread_id * CACHED_VERTICES * CACHE_STEP];
    return private_data;
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
