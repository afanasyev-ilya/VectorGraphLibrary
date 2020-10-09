#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
class VerticesArray
{
public:
    TraversalDirection direction;
    VectCSRGraph *graph_ptr;

    _T *vertices_data;
    int vertices_count;
public:
    /* constructors and destructors */
    VerticesArray() {};
    ~VerticesArray() {};

    /* get/set API */
    _T *get_ptr() {return vertices_data;};

    /* direction API */
    TraversalDirection get_direction() {return direction;};
    void set_direction(TraversalDirection _direction) {direction = _direction;};

    /* initialization API */
    void set_all_constant(_T _const) {};
    void set_all_random(_T _max_rand) {};

    /* print API */
    void print();
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "nec/vertices_array/vertices_array_nec.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
