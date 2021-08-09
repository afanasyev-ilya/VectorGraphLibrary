#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename _TIndex>
void openmp_reorder_gather_inplace(_T *_result, _T *_buffer, _TIndex *_indexes, size_t _size)
{
    if(omp_in_parallel())
    {
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vector
        #pragma omp for
        for(_TIndex i = 0; i < _size; i++)
        {
            _buffer[i] = _result[_indexes[i]];
        }

        #pragma _NEC ivdep
        #pragma omp for
        for(_TIndex i = 0; i < _size; i++)
        {
            _result[i] = _buffer[i];
        }
    }
    else
    {
        #pragma omp parallel
        {
            openmp_reorder_gather_inplace(_result, _buffer, _indexes, _size);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename _TIndex>
void openmp_reorder_gather_copy(_T *_gather_from, _T *_output, _TIndex *_indexes, size_t _size)
{
    if(omp_in_parallel())
    {
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vector
        #pragma omp for
        for (_TIndex i = 0; i < _size; i++)
        {
            _output[i] = _gather_from[_indexes[i]];
        }
    }
    else
    {
        #pragma omp parallel
        {
            openmp_reorder_gather_copy(_gather_from, _output, _indexes, _size);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename _TIndex>
void openmp_reorder_scatter_inplace(_T *_result, _T *_buffer, _TIndex *_indexes, size_t _size)
{
    if(omp_in_parallel())
    {
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vector
        #pragma omp for
        for(_TIndex i = 0; i < _size; i++)
        {
            _buffer[_indexes[i]] = _result[i];
        }

        #pragma _NEC ivdep
        #pragma omp for
        for(_TIndex i = 0; i < _size; i++)
        {
            _result[i] = _buffer[i];
        }
    }
    else
    {
        #pragma omp parallel
        {
            openmp_reorder_scatter_inplace(_result, _buffer, _indexes, _size);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename _TIndex>
void openmp_reorder_scatter_copy(_T *_scatter_from, _T *_output, _TIndex *_indexes, size_t _size)
{
    if(omp_in_parallel())
    {
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vector
        #pragma omp for
        for(_TIndex i = 0; i < _size; i++)
        {
            _output[_indexes[i]] = _scatter_from[i];
        }
    }
    else
    {
        #pragma omp parallel
        {
            openmp_reorder_scatter_inplace(_scatter_from, _output, _indexes, _size);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
