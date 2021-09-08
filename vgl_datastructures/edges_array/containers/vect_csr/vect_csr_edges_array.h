#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
class EdgesArray_VectorCSR : public BaseEdgesArray<_T>
{
private:
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

    /* initialization API */
    void set_all_constant(_T _const) final;
    void set_all_random(_T _max_rand) final;

    /* print API */
    void print() final;

    void attach_pointer(_T *_outer_data) final;

    template <typename MergeOperation>
    void finalize_advance(MergeOperation &&merge_operation);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "vect_csr_edges_array.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
