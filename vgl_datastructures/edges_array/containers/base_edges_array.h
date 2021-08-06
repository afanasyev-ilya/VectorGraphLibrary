#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
class BaseEdgesArray
{
protected:
    VGL_Graph *graph_ptr;

    _T *edges_data;
    long long edges_count;
    long long total_array_size;

    bool is_copy;
public:
    /* constructors and destructors */
    BaseEdgesArray() {};
    ~BaseEdgesArray() {};

    /* get/set API */
    long long get_total_array_size() { return total_array_size; };

    /* initialization API */
    virtual void set_all_constant(_T _const) = 0;
    virtual void set_all_random(_T _max_rand) = 0;

    /* print API */
    virtual void print() = 0;

    virtual void attach_pointer(_T *_outer_data) = 0;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "csr/csr_edges_array.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
