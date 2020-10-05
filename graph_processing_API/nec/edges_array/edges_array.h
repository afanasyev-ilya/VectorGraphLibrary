#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
class EdgesArrayNec
{
private:
    _T *edges_data;

    _T *outgoing_csr_ptr;
    _T *incoming_csr_ptr;

    _T *outgoing_ve_ptr;
    _T *incoming_ve_ptr;

    long long edges_count;
    long long edges_count_in_outgoing_ve;
    long long edges_count_in_incoming_ve;
    long long wall_array_size;
public:
    template <typename _TVertexValue, typename _TEdgeWeight>
    EdgesArrayNec(VectCSRGraph<_TVertexValue, _TEdgeWeight> &_graph);

    _T get(long long _global_idx) {return edges_data[_global_idx];};
    _T set(long long _global_idx, _T _val) {edges_data[_global_idx] = _val;};

    ~EdgesArrayNec();
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// how to set all edges random?

// 2 подхода
// 1) из Advance передавать нужный edges_pos для разных случаев (CSR, VE, push/pull)
// в lambda тогда будет weights.get(edge_pos); - всегда брать верный элемент
// как это сделать в Advance? тип (VE/CSR) - знаем, направление (какой массив) - знаем - можно посчитать
// как пользоателю заполнять массив? допустим нужно задать вес src_id -> dst_id
// вот это нужно в первую очередь
// edges_array_size = edges_count + edges_count_in_ve;
// traversal = 0/1
// storage = 0/1
// csr_edge_pos = pos/0
// ve_edge_pos = pos/0
// long long idx = traversal_type * edges_array_size + storage_type * edges_count + csr_edge_pos + ve_edge_pos
// weights[idx] - то, что надо
// проверить будет ли векторизовать??? на глупых массивах

// set(_graph, src, dst, val);
// сможем ли реализовать такой? - конечно! смотрим direction (есть ли он в графе), делаем 2 записи в ve и csr
// обращения внутри Advance будут - хорошими, либо в CSR, либо в VE

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "edges_array.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
