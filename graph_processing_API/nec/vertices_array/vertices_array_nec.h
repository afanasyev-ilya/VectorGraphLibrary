#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
class VerticesArrayNec : VerticesArray<_T>
{
private:
    TraversalDirection direction;

    VectCSRGraph *graph_ptr;

    _T *vertices_data;
    int vertices_count;
public:
    /* constructors and destructors */
    VerticesArrayNec(VectCSRGraph &_graph, TraversalDirection _direction = SCATTER);
    VerticesArrayNec(const VerticesArrayNec<_T> &_copy_obj);
    ~VerticesArrayNec();

    /* get/set API */
    inline _T get(int _idx) {return vertices_data[_idx];};
    inline _T set(int _idx, _T _val) {vertices_data[_idx] = _val;};

    // get/set operator (for both const/non const versions)
    inline _T& operator[](int _idx) { return vertices_data[_idx]; }
    const inline _T& operator[] (int _idx) const { return vertices_data[_idx]; };

    // get pointer to data array
    _T *get_ptr() {return vertices_data;};

    /* direction API */
    TraversalDirection get_direction() {return direction;};
    void set_direction(TraversalDirection _direction) {direction = _direction;};

    /* initialization API */
    void set_all_constant(_T _const);
    void set_all_random(_T _max_rand);

    /* print API */
    void print();
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "vertices_array_nec.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
