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
        _TIndex sorted_index = _indexes[i];
        _TIndex original_index = i;
        _data_buffer[original_index] = _data[sorted_index];
    }

    #pragma _NEC ivdep
    #pragma omp for
    for(_TIndex i = 0; i < _size; i++)
    {
        _data[i] = _data_buffer[i];
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
