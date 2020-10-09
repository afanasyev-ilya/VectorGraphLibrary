#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../../../common/memory_API/memory_API.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class VectorExtension
{
private:
    int vertices_count;
    int starting_vertex;
    int vector_segments_count;
    long long edges_count_in_ve;

    int first_vertex, last_vertex;

    // vector segments data
    long long *vector_group_ptrs;
    int *vector_group_sizes;

    // CSR ptrs required for fast conversions
    long long *csr_adjacent_ptrs_ptr;

    // outgoing edges data
    int *adjacent_ids;

    void alloc(long long _edges_count);
    void free();
public:
    VectorExtension();
    ~VectorExtension();

    inline int get_vertices_count() {return vertices_count;};
    inline int get_starting_vertex() {return starting_vertex;};
    inline int get_vector_segments_count() {return vector_segments_count;};

    inline long long get_edges_count_in_ve() {return edges_count_in_ve;};

    long long *get_vector_group_ptrs() {return vector_group_ptrs;};
    int *get_vector_group_sizes() {return vector_group_sizes;};
    int *get_adjacent_ids() {return adjacent_ids;};

    void init_from_graph(long long *_csr_adjacent_ptrs, int *_csr_adjacent_ids,
                         int _first_vertex, int _last_vertex);

    inline long long get_ve_edge_id(int _src_id, int _dst_id);

    template <typename _T>
    void copy_array_from_csr_to_ve(_T *_dst_ve_array, _T *_src_csr_array);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "vector_extension.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
