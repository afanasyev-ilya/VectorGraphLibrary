#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class MemoryAPI
{
public:
    template <typename _T>
    static void allocate_array(_T **_ptr, size_t _size);

    template <typename _T>
    static void free_array(_T *_ptr);

    #ifdef __USE_GPU__
    template <typename _T>
    static void allocate_host_array(_T **_ptr, size_t _size) {allocate_array(_ptr, _size);};

    template <typename _T>
    static void free_host_array(_T *_ptr) {free_array(_ptr);};

    template <typename _T>
    static void allocate_device_array(_T **_ptr, size_t _size);

    template <typename _T>
    static void free_device_array(_T *_ptr);

    template <typename _T>
    static void move_array_to_device(_T **_ptr, size_t _size);

    template <typename _T>
    static void move_array_to_host(_T **_ptr, size_t _size);
    #endif
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "memory_API.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

