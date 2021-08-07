#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
class EdgesArray_VectorCSR : public BaseEdgesArray<_T>
{
private:
    _T *outgoing_edges;
    _T *incoming_edges;

    _T *outgoing_edges_ve;
    _T *incoming_edges_ve;

    long long edges_count_in_outgoing_csr;
    long long edges_count_in_incoming_csr;
    long long edges_count_in_outgoing_ve;
    long long edges_count_in_incoming_ve;
public:
    /* constructors and destructors */
    EdgesArray_VectorCSR(VGL_Graph &_graph);
    ~EdgesArray_VectorCSR();

    /* get/set API */
    inline void set(int _src_id, int _dst_id, _T _val, TraversalDirection _direction);
    inline _T get(int _src_id, int _dst_id, TraversalDirection _direction);

    /* initialization API */
    void set_all_constant(_T _const);
    void set_all_random(_T _max_rand);

    /* print API */
    void print();

    void attach_pointer(_T *_outer_data) final;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "vect_csr_edges_array.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
