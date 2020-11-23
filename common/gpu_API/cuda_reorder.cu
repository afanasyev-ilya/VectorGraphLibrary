#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void __global__ reorder_kernel(_T *_data, _T *_data_buffer, int *_indexes, int _size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < _size)
    {
        int sorted_index = idx;
        int original_index = _indexes[idx];
        _data_buffer[_indexes[idx]] = _data[idx];
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void __global__ copy_kernel(_T *_data, _T *_data_buffer, int _size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < _size)
    {
        _data[idx] = _data_buffer[idx];
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void cuda_reorder_wrapper(_T *_data, _T *_data_buffer, int *_indexes, int _size)
{
    SAFE_KERNEL_CALL((reorder_kernel<<<(_size - 1)/BLOCK_SIZE + 1, BLOCK_SIZE>>>(_data, _data_buffer, _indexes, _size)));
    SAFE_KERNEL_CALL((copy_kernel<<<(_size - 1)/BLOCK_SIZE + 1, BLOCK_SIZE>>>(_data, _data_buffer, _size)));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template void cuda_reorder_wrapper<int>(int *_data, int *_data_buffer, int *_indexes, int _size);
template void cuda_reorder_wrapper<float>(float *_data, float *_data_buffer, int *_indexes, int _size);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
