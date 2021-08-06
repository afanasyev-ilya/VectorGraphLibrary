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
    void set_all_constant(_T _const);
    void set_all_random(_T _max_rand);
    void set_equal_to_index();

    /* print API */
    void print();
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "csr_edges_array.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
