#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>

#ifdef __USE_GPU__
#include <cuda_runtime.h>
#include "../gpu_API/cuda_error_handling.h"
#endif

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
    static void allocate_host_array(_T **_ptr, size_t _size) {allocate_array(_ptr, _size);};

    template <typename _T>
    static void free_host_array(_T *_ptr) {free_array(_ptr);};

    template <typename _T>
    static void allocate_device_array(_T **_ptr, size_t _size);

    template <typename _T>
    static void allocate_non_managed_array(_T **_ptr, size_t _size);

    template <typename _T>
    static void allocate_managed_array(_T **_ptr, size_t _size);

    template <typename _T>
    static void prefetch_managed_array(_T *_ptr, size_t _size);

    template <typename _T>
    static void free_device_array(_T *_ptr);

    template <typename _T>
    static void move_array_to_device(_T **_ptr, size_t _size);

    template <typename _T>
    static void move_array_to_host(_T **_ptr, size_t _size);

    template <typename _T>
    static void copy_array_to_device(_T *_dst, _T *_src, size_t _size);

    template <typename _T>
    static void copy_array_to_host(_T *_dst, _T *_src, size_t _size);
    #endif
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef __CUDA_INCLUDE__
#include "memory_API.hpp"
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

