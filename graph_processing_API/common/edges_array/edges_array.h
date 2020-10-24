#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
class EdgesArray
{
protected:
    BaseGraph *graph_ptr;

    _T *edges_data;
    long long edges_count;
    long long total_array_size;

    bool is_copy;
public:
    /* constructors and destructors */
    EdgesArray() {};
    ~EdgesArray() {};

    /* get/set API */
    #ifdef __USE_GPU__
    __host__ __device__ inline _T get(long long _global_idx) const {return edges_data[_global_idx];};
    __host__ __device__ inline _T set(long long _global_idx, _T _val) const {edges_data[_global_idx] = _val;};
    __host__ __device__ inline _T& operator[] (long long _global_idx) const { return edges_data[_global_idx]; };
    #endif

    #if defined(__USE_NEC_SX_AURORA__) || defined(__USE_INTEL__)
    inline _T get(long long _global_idx) const {return edges_data[_global_idx];};
    inline void set(long long _global_idx, _T _val) const {edges_data[_global_idx] = _val;};
    inline _T& operator[] (long long _global_idx) const { return edges_data[_global_idx]; };
    #endif

    inline _T *get_ptr() const {return edges_data;};

    /* initialization API */
    virtual void set_all_constant(_T _const) = 0;
    virtual void set_all_random(_T _max_rand) = 0;

    /* print API */
    virtual void print() = 0;

    /* GPU specific (copy) API */
    #ifdef __USE_GPU__
    void move_to_device();
    void move_to_host();
    #endif
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "gpu_api.hpp"
#include "el/edges_array.h"
#include "vect/edges_array.h"
#include "sharded/edges_array.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
