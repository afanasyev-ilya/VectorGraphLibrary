#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
class EdgesArray
{
private:
    VectCSRGraph *graph_ptr;

    _T *edges_data;

    _T *outgoing_csr_ptr;
    _T *incoming_csr_ptr;

    _T *outgoing_ve_ptr;
    _T *incoming_ve_ptr;

    long long edges_count;
    long long edges_count_in_outgoing_ve;
    long long edges_count_in_incoming_ve;
    long long total_array_size;
public:
    /* constructors and destructors */
    EdgesArray(VectCSRGraph &_graph);
    ~EdgesArray();

    /* get/set API */
    inline _T get(long long _global_idx) {return edges_data[_global_idx];};
    inline _T set(long long _global_idx, _T _val) {edges_data[_global_idx] = _val;};

    void set(int _src_id, int _dst_id, _T _val, TraversalDirection _direction);
    _T get(int _src_id, int _dst_id, TraversalDirection _direction);

    /* initialization API */
    void set_all_constant(_T _const);
    void set_all_random(_T _max_rand);

    /* print API */
    void print();

    /* GPU specific (copy) API */
    #ifdef __USE_GPU__
    void move_to_device();
    void move_to_host();
    #endif
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef __CUDA_INCLUDE__
#include "edges_array.hpp"
#include "gpu_api.hpp"
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
