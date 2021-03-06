#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
class EdgesArray_EL : public EdgesArray<_T>
{
public:
    /* constructors and destructors */
    EdgesArray_EL(EdgesListGraph &_graph);
    EdgesArray_EL(const EdgesArray_EL<_T> &_copy_obj);
    ~EdgesArray_EL();

    /* initialization API */
    void set_all_constant(_T _const);
    void set_all_random(_T _max_rand);
    void set_equal_to_index();

    /* print API */
    void print();
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "edges_array.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
