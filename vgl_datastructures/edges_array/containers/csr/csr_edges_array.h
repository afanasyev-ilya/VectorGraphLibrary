#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
class EdgesArray_CSR : public BaseEdgesArray<_T>
{
public:
    /* constructors and destructors */
    EdgesArray_CSR(VGL_Graph &_graph);
    ~EdgesArray_CSR();

    /* initialization API */
    void set_all_constant(_T _const) final;
    void set_all_random(_T _max_rand) final;

    /* print API */
    void print()  final;

    void attach_pointer(_T *_outer_data) final;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "csr_edges_array.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
