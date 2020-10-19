#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class MemoryAPI
{
public:
    template <typename _T>
    static void allocate_array(_T **_ptr, size_t _size);

    template <typename _T>
    static void free_array(_T *_ptr);

    template <typename _T>
    static void copy(_T *_dst, _T *_src, size_t _size);

    template <typename _T>
    static void set(_T *_data, _T val, size_t _size);

    #ifdef __USE_GPU__
    template <typename _T>
    static void move_array_to_device(_T *_ptr, size_t _size);

    template <typename _T>
    static void move_array_to_host(_T *_ptr, size_t _size);
    #endif
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef __CUDA_INCLUDE__
#include "memory_API.hpp"
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

