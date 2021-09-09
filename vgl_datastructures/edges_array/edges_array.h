#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "containers/base_edges_array.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
class EdgesArray
{
private:
    ObjectType object_type; // EDGES_ARRAY

    _T *edges_data;

    BaseEdgesArray<_T> *container;
    bool is_copy;
public:
    /* constructors and destructors */
    EdgesArray(VGL_Graph &_graph);
    EdgesArray(const EdgesArray<_T> &_copy_obj);
    ~EdgesArray();

    // the following getters and setters are implemented here, since calling virtual functions of edges_array_representation
    // is too slow for graph processing (reduces bandwidth on NEC and CPUs)
    // luckily, all _global_idx indexes are calculated inside VGL advance abstraction

    /* get/set API */
    #ifdef __USE_GPU__
    __host__ __device__ inline _T get(long long _global_idx) const {return edges_data[_global_idx];};
    __host__ __device__ inline _T set(long long _global_idx, _T _val) const {edges_data[_global_idx] = _val;};
    __host__ __device__ inline _T& operator[] (long long _global_idx) const { return edges_data[_global_idx]; };
    #else
    inline _T get(long long _global_idx) const {return edges_data[_global_idx];};
    inline void set(long long _global_idx, _T _val) const {edges_data[_global_idx] = _val;};
    inline _T& operator[] (long long _global_idx) const { return edges_data[_global_idx]; };
    #endif

    inline _T *get_ptr() const { return edges_data; };

    /* initialization API */
    void set_all_constant(_T _const) { container->set_all_constant(_const); };
    void set_all_random(_T _max_rand) { container->set_all_random(_max_rand); };

    /* print API */
    void print() { container->print(); };

    /* GPU specific (copy) API */
    #ifdef __USE_GPU__
    void move_to_device();
    void move_to_host();
    #endif

    template <typename MergeOperation>
    void finalize_advance(MergeOperation &&merge_operation);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "edges_array.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
