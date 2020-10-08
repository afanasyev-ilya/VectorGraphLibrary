/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void MemoryAPI::allocate_array(_T **_ptr, size_t _size)
{
    #if defined(__USE_NEC_SX_AURORA__)
    *_ptr = (_T*)aligned_alloc(sizeof(_T), _size*sizeof(_T));
    #elif defined(__USE_GPU__)
    *_ptr = (_T*)malloc(_size*sizeof(_T));
    #else
    *_ptr = (_T*)malloc(_size*sizeof(_T));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void MemoryAPI::free_array(_T *_ptr)
{
    #if defined(__USE_NEC_SX_AURORA__)
    free(_ptr);
    #elif defined(__USE_GPU__)
    free(_ptr);
    #else
    free(_ptr);
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void MemoryAPI::copy(_T *_dst, _T *_src, size_t _size)
{
    #pragma _NEC ivdep
    #pragma omp parallel
    for(long long i = 0; i < _size; i++)
    {
        _dst[i] = _src[i];
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void MemoryAPI::set(_T *_data, _T _val, size_t _size)
{
    #pragma _NEC ivdep
    #pragma omp parallel for
    for(long long i = 0; i < _size; i++)
    {
        _data[i] = _val;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _T>
void MemoryAPI::allocate_device_array(_T **_ptr, size_t _size)
{
    #ifdef __USE_MANAGED_MEMORY__
    SAFE_CALL(cudaMallocManaged((void**)_ptr, _size * sizeof(_T)));
    #else
    SAFE_CALL(cudaMalloc((void**)_ptr, _size * sizeof(_T)));
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _T>
void MemoryAPI::allocate_non_managed_array(_T **_ptr, size_t _size)
{
    SAFE_CALL(cudaMalloc((void**)_ptr, _size * sizeof(_T)));
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _T>
void MemoryAPI::prefetch_managed_array(_T *_ptr, size_t _size)
{
    int device = -1;
    SAFE_CALL(cudaGetDevice(&device));
    SAFE_CALL(cudaMemPrefetchAsync(_ptr, _size*sizeof(_T), device, NULL));
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _T>
void MemoryAPI::allocate_managed_array(_T **_ptr, size_t _size)
{
    SAFE_CALL(cudaMallocManaged((void**)_ptr, _size * sizeof(_T)));
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _T>
void MemoryAPI::free_device_array(_T *_ptr)
{
    SAFE_CALL(cudaFree((void*)_ptr));
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _T>
void MemoryAPI::move_array_to_device(_T **_ptr, size_t _size)
{
    _T *host_ptr = *_ptr;
    _T *device_ptr;

    MemoryAPI::allocate_device_array(&device_ptr, _size);
    SAFE_CALL(cudaMemcpy(device_ptr, host_ptr, _size * sizeof(_T), cudaMemcpyHostToDevice));

    MemoryAPI::free_host_array(host_ptr);
    *_ptr = device_ptr;
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _T>
void MemoryAPI::move_array_to_host(_T **_ptr, size_t _size)
{
    _T *host_ptr;
    MemoryAPI::allocate_host_array(&host_ptr, _size);
    _T *device_ptr = *_ptr;

    SAFE_CALL(cudaMemcpy(host_ptr, device_ptr, _size * sizeof(_T), cudaMemcpyDeviceToHost));
    MemoryAPI::free_device_array(device_ptr);

    *_ptr = host_ptr;
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _T>
void MemoryAPI::copy_array_to_device(_T *_dst, _T *_src, size_t _size)
{
    SAFE_CALL(cudaMemcpy(_dst, _src, _size * sizeof(_T), cudaMemcpyHostToDevice));
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _T>
void MemoryAPI::copy_array_to_host(_T *_dst, _T *_src, size_t _size)
{
    SAFE_CALL(cudaMemcpy(_dst, _src, _size * sizeof(_T), cudaMemcpyDeviceToHost));
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

