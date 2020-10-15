#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
class VerticesArrayNEC : public VerticesArray<_T>
{
public:
    /* constructors and destructors */
    VerticesArrayNEC(VectCSRGraph &_graph, TraversalDirection _direction = SCATTER);
    VerticesArrayNEC(const VerticesArrayNEC<_T> &_copy_obj);
    ~VerticesArrayNEC();

    /* get/set API */
    inline _T get(int _idx) {return this->vertices_data[_idx];};
    inline _T set(int _idx, _T _val) {this->vertices_data[_idx] = _val;};

    // get/set operator (for both const/non const versions)
    inline _T& operator[](int _idx) { return this->vertices_data[_idx]; }
    const inline _T& operator[] (int _idx) const { return this->vertices_data[_idx]; };

    /* initialization API */
    void set_all_constant(_T _const);
    void set_all_random(_T _max_rand);

    /* print API */
    void print();
    void print(string _name);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "vertices_array_nec.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
