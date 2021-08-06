#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename _TIndex>
void __global__ reorder_kernel_scatter(_T *_data, _T *_data_buffer, _TIndex *_indexes, _TIndex _size)
{
    const _TIndex idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < _size)
    {
        _TIndex sorted_index = idx;
        _TIndex original_index = _indexes[idx];
        _data_buffer[_indexes[idx]] = _data[idx];
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename _TIndex>
void __global__ reorder_kernel_gather_copy(_T *_dst, _T *_src, _TIndex *_indexes, _TIndex _size)
{
    const _TIndex idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < _size)
    {
        _dst[idx] = _src[_indexes[idx]];
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename _TIndex>
void __global__ copy_kernel(_T *_data, _T *_data_buffer, _TIndex _size)
{
    const _TIndex idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < _size)
    {
        _data[idx] = _data_buffer[idx];
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename _TIndex>
void cuda_reorder_wrapper_scatter(_T *_data, _T *_data_buffer, _TIndex *_indexes, _TIndex _size)
{
    SAFE_KERNEL_CALL((reorder_kernel_scatter<<<(_size - 1)/BLOCK_SIZE + 1, BLOCK_SIZE>>>(_data, _data_buffer, _indexes, _size)));
    SAFE_KERNEL_CALL((copy_kernel<<<(_size - 1)/BLOCK_SIZE + 1, BLOCK_SIZE>>>(_data, _data_buffer, _size)));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename _TIndex>
void cuda_reorder_wrapper_gather_copy(_T *_dst, _T *_src, _TIndex *_indexes, _TIndex _size)
{
    SAFE_KERNEL_CALL((reorder_kernel_gather_copy<<<(_size - 1)/BLOCK_SIZE + 1, BLOCK_SIZE>>>(_dst, _src, _indexes, _size)));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////