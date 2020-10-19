#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
class VerticesArray
{
private:
    ObjectType object_type;
    TraversalDirection direction;
    VectCSRGraph *graph_ptr;

    _T *vertices_data;
    int vertices_count;
public:
    /* constructors and destructors */
    VerticesArray(VectCSRGraph &_graph, TraversalDirection _direction = SCATTER);
    VerticesArray(const VerticesArray<_T> &_copy_obj);
    ~VerticesArray();

    /* get/set API */
    _T *get_ptr() {return vertices_data;};
    ObjectType get_object_type() {return object_type;};

    inline _T get(int _idx) {return this->vertices_data[_idx];};
    inline _T set(int _idx, _T _val) {this->vertices_data[_idx] = _val;};

    inline _T& operator[](int _idx) { return vertices_data[_idx]; }
    const inline _T& operator[] (int _idx) const { return vertices_data[_idx]; };

    /* direction API */
    TraversalDirection get_direction() {return direction;};
    void set_direction(TraversalDirection _direction) {direction = _direction;};

    /* initialization API */
    void set_all_constant(_T _const);
    void set_all_random(_T _max_rand);

    /* print API */
    void print();
    void print(string _name);

    /* GPU specific (copy) API */
    #ifdef __USE_GPU__
    void move_to_device();
    void move_to_host();
    #endif
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template class VerticesArray<int>;
template class VerticesArray<float>;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef __CUDA_INCLUDE__
#include "vertices_array.hpp"
#include "gpu_api.hpp"
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
