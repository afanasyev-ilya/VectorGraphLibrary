#pragma once


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
class VerticesArrayNec
{
private:
    TraversalDirection direction;

    VectCSRGraph *graph_ptr;

    _T *vertices_data;
    int vertices_count;
public:
    VerticesArrayNec(VectCSRGraph &_graph, TraversalDirection _direction = SCATTER);

    VerticesArrayNec(const VerticesArrayNec<_T> &_copy_obj);

    inline _T get(int _idx) {return vertices_data[_idx];};
    inline _T set(int _idx, _T _val) {vertices_data[_idx] = _val;};

    inline _T& operator[](int _idx) { return vertices_data[_idx]; }
    const inline _T& operator[] (int _idx) const { return vertices_data[_idx]; };

    TraversalDirection get_direction() {return direction;};
    void set_direction(TraversalDirection _direction) {direction = _direction;};

    _T *get_ptr() {return vertices_data;};

    void set_all_constant(_T _const);
    void set_all_random(_T _max_rand);

    void print();

    ~VerticesArrayNec();
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "vertices_array_nec.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
