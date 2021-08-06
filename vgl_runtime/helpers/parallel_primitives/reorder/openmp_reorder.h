#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename _TIndex>
void openmp_reorder_wrapper_gather(_T *_data, _T *_data_buffer, _TIndex *_indexes, _TIndex _size)
{
    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma omp for
    for(_TIndex i = 0; i < _size; i++)
    {
        _data_buffer[i] = _data[_indexes[i]];
    }

    #pragma _NEC ivdep
    #pragma omp for
    for(_TIndex i = 0; i < _size; i++)
    {
        _data[i] = _data_buffer[i];
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename _TIndex>
void openmp_reorder_wrapper_gather_inplace(_T *_gather_from, _T *_output, _TIndex *_indexes, _TIndex _size)
{
    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma omp for
    for(_TIndex i = 0; i < _size; i++)
    {
        _output[i] = _gather_from[_indexes[i]];
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename _TIndex>
void openmp_reorder_wrapper_scatter(_T *_data, _T *_data_buffer, _TIndex *_indexes, _TIndex _size)
{
    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma omp for
    for(_TIndex i = 0; i < _size; i++)
    {
        _data_buffer[_indexes[i]] = _data[i];
    }

    #pragma _NEC ivdep
    #pragma omp for
    for(_TIndex i = 0; i < _size; i++)
    {
        _data[i] = _data_buffer[i];
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename _TIndex>
void openmp_reorder_wrapper_scatter_copy(_T *_in_data, _T *_out_data, _TIndex *_indexes, _TIndex _size)
{
    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma omp for
    for(_TIndex i = 0; i < _size; i++)
    {
        _out_data[_indexes[i]] = _in_data[i];
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
