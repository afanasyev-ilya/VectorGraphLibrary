#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
class EdgesArray_Vect : public EdgesArray<_T>
{
private:
    _T *outgoing_csr_ptr;
    _T *incoming_csr_ptr;

    _T *outgoing_ve_ptr;
    _T *incoming_ve_ptr;

    long long edges_count_in_outgoing_csr;
    long long edges_count_in_incoming_csr;
    long long edges_count_in_outgoing_ve;
    long long edges_count_in_incoming_ve;
public:
    /* constructors and destructors */
    EdgesArray_Vect(VectCSRGraph &_graph);
    EdgesArray_Vect(const EdgesArray_Vect<_T> &_copy_obj);
    ~EdgesArray_Vect();

    /* get/set API */
    inline void set(int _src_id, int _dst_id, _T _val, TraversalDirection _direction);
    inline _T get(int _src_id, int _dst_id, TraversalDirection _direction);

    /* initialization API */
    void set_all_constant(_T _const);
    void set_all_random(_T _max_rand);

    /* print API */
    void print();

    /* remaining API */
    void operator = (const EdgesArray_EL<_T> &_el_data);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "edges_array.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
