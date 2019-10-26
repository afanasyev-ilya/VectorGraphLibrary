//
//  programming_API.hpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 06/05/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::set_threads_count(int _threads_count)
{
    threads_count = _threads_count;
    
    #pragma omp parallel for num_threads(threads_count)
    for(int i = 0; i < threads_count; i++);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
template <class _T>
_T* ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::vertex_array_alloc()
{
    _T *new_ptr = new _T[this->vertices_count];
    return new_ptr;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
template <class _T>
void ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::vertex_array_copy(_T *_dst_array, _T *_src_array)
{
    int vertices_count = this->vertices_count;
    #pragma omp parallel for num_threads(threads_count)
    for(int i = 0; i < vertices_count; i++)
        _dst_array[i] = _src_array[i];
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
template <class _T>
void ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::vertex_array_set_to_constant(_T *_dst_array, _T _value)
{
    int vertices_count = this->vertices_count;
    #pragma omp parallel for num_threads(threads_count)
    for(int i = 0; i < vertices_count; i++)
        _dst_array[i] = _value;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
template <class _T>
void ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::vertex_array_set_element(_T *_dst_array, int _pos, _T _value)
{
    _dst_array[_pos] = _value;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
template <class _T>
_T* ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::allocate_private_caches(int _threads_count)
{
    _T *new_ptr = new _T[_threads_count * CACHED_VERTICES * CACHE_STEP];
    #ifdef __USE_NEC_SX_AURORA__
    #pragma retain(new_ptr)
    #endif
    return new_ptr;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
template <class _T>
void ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::free_data(_T *_array)
{
    delete []_array;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
template <class _T>
inline _T ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::load_vertex_data_cached(int _idx, _T *_data, _T *_private_data)
{
    _T result = 0;
    if(_idx < CACHED_VERTICES)
        result = _private_data[_idx * CACHE_STEP];
    else
        result = _data[_idx];
    return result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
template <class _T>
inline _T ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::load_vertex_data(int _idx, _T *_data)
{
    return _data[_idx];
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
template <class _T>
inline _T ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::place_data_into_cache(_T *_data, _T *_private_data)
{
    #ifdef __USE_NEC_SX_AURORA__
    #pragma _NEC vector
    #endif
    for(int i = 0; i < CACHED_VERTICES; i++)
        _private_data[i * CACHE_STEP] = _data[i];
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
template <class _T>
inline _T* ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::get_private_data_pointer(_T *_cached_data)
{
    int thread_id = omp_get_thread_num();
    _T *private_data = &_cached_data[thread_id * CACHED_VERTICES * CACHE_STEP];
    return private_data;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
