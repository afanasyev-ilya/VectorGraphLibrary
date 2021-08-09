#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename _TIndex>
void cuda_reorder_gather(_T *_data, _T *_data_buffer, _TIndex *_indexes, size_t _size);

template <typename _T, typename _TIndex>
void cuda_reorder_scatter(_T *_data, _T *_data_buffer, _TIndex *_indexes, size_t _size);

template <typename _T, typename _TIndex>
void cuda_reorder_gather_inplace(_T *_data, _T *_data_buffer, _TIndex *_indexes, size_t _size);

template <typename _T, typename _TIndex>
void cuda_reorder_scatter_inplace(_T *_data, _T *_data_buffer, _TIndex *_indexes, size_t _size);

template <typename _T, typename _TIndex>
void cuda_reorder_gather_copy(_T *_input, _T *_output, _TIndex *_indexes, size_t _size);

template <typename _T, typename _TIndex>
void cuda_reorder_scatter_copy(_T *_input, _T *_output, _TIndex *_indexes, size_t _size);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "cuda_reorder.cu"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
