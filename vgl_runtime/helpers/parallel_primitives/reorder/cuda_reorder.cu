#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename _TIndex>
void __global__ reorder_kernel_scatter(_T *_data, _T *_data_buffer, _TIndex *_indexes, size_t _size)
{
    const _TIndex idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < _size)
    {
        _data_buffer[_indexes[idx]] = _data[idx];
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename _TIndex>
void __global__ reorder_kernel_gather(_T *_data, _T *_data_buffer, _TIndex *_indexes, size_t _size)
{
    const _TIndex idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < _size)
    {
        _data_buffer[idx] = _data[_indexes[idx]];
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename _TIndex>
void __global__ copy_kernel(_T *_dst, _T *_src, _TIndex _size)
{
    const _TIndex idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < _size)
    {
        _dst[idx] = _src[idx];
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename _TIndex>
void cuda_reorder_gather_inplace(_T *_data, _T *_data_buffer, _TIndex *_indexes, size_t _size)
{
    SAFE_KERNEL_CALL((reorder_kernel_gather<<<(_size - 1)/BLOCK_SIZE + 1, BLOCK_SIZE>>>(_data, _data_buffer, _indexes, _size)));
    SAFE_KERNEL_CALL((copy_kernel<<<(_size - 1)/BLOCK_SIZE + 1, BLOCK_SIZE>>>(_data, _data_buffer, _size)));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename _TIndex>
void cuda_reorder_scatter_inplace(_T *_data, _T *_data_buffer, _TIndex *_indexes, size_t _size)
{
    SAFE_KERNEL_CALL((reorder_kernel_scatter<<<(_size - 1)/BLOCK_SIZE + 1, BLOCK_SIZE>>>(_data, _data_buffer, _indexes, _size)));
    SAFE_KERNEL_CALL((copy_kernel<<<(_size - 1)/BLOCK_SIZE + 1, BLOCK_SIZE>>>(_data, _data_buffer, _size)));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename _TIndex>
void cuda_reorder_gather_copy(_T *_input, _T *_output, _TIndex *_indexes, size_t _size)
{
    SAFE_KERNEL_CALL((reorder_kernel_gather<<<(_size - 1)/BLOCK_SIZE + 1, BLOCK_SIZE>>>(_input, _output, _indexes, _size)));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename _TIndex>
void cuda_reorder_scatter_copy(_T *_input, _T *_output, _TIndex *_indexes, size_t _size)
{
    SAFE_KERNEL_CALL((reorder_kernel_scatter<<<(_size - 1)/BLOCK_SIZE + 1, BLOCK_SIZE>>>(_input, _output, _indexes, _size)));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
