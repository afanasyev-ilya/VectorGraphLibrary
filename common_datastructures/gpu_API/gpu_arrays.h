//
//  move_arrays.h
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 01/05/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef gpu_arrays_h
#define gpu_arrays_h

#ifdef __USE_GPU__

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "cuda_error_handling.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void move_array_to_device(_T **_ptr, long long int _size)
{
    _T *host_ptr = *_ptr;
    _T *device_ptr;
    
    SAFE_CALL(cudaMalloc((void**)&device_ptr, _size * sizeof(_T)));
    SAFE_CALL(cudaMemcpy(device_ptr, host_ptr, _size * sizeof(_T), cudaMemcpyHostToDevice));
              
    delete []host_ptr;
    
    *_ptr = device_ptr;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void move_array_to_host(_T **_ptr, long long int _size)
{
    _T *host_ptr = new _T[_size];
    _T *device_ptr = *_ptr;
    
    SAFE_CALL(cudaMemcpy(host_ptr, device_ptr, _size * sizeof(_T), cudaMemcpyDeviceToHost));
    SAFE_CALL(cudaFree(device_ptr));
    
    *_ptr = host_ptr;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* __USE_GPU__ */

#endif /* gpu_arrays_h */
