#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight, typename _T>
class EdgesArrayNec
{
private:
    VectCSRGraph<_TVertexValue, _TEdgeWeight> *graph_ptr;

    _T *edges_data;

    _T *outgoing_csr_ptr;
    _T *incoming_csr_ptr;

    _T *outgoing_ve_ptr;
    _T *incoming_ve_ptr;

    long long edges_count;
    long long edges_count_in_outgoing_ve;
    long long edges_count_in_incoming_ve;
    long long wall_array_size;
public:
    EdgesArrayNec(VectCSRGraph<_TVertexValue, _TEdgeWeight> &_graph);

    inline _T get(long long _global_idx) {return edges_data[_global_idx];};
    inline _T set(long long _global_idx, _T _val) {edges_data[_global_idx] = _val;};

    void set_all_constant(_T _const);
    void set_all_random(_T _max_rand);

    ~EdgesArrayNec();
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "edges_array.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
