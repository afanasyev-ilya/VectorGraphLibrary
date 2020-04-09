#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../../../common/memory_API/memory_API.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
class VectorExtension
{
private:
    int vertices_count;
    int starting_vertex;
    int vector_segments_count;

    // vector segments data
    long long *vector_group_ptrs;
    int *vector_group_sizes;

    // outgoing edges data
    int *adjacent_ids;
    _TEdgeWeight *adjacent_weights;

    void alloc(long long _edges_count);
    void free();
public:
    VectorExtension();
    ~VectorExtension();

    int get_vertices_count() {return vertices_count;};
    int get_starting_vertex() {return starting_vertex;};
    int get_vector_segments_count() {return vector_segments_count;};

    long long *get_vector_group_ptrs() {return vector_group_ptrs;};
    int *get_vector_group_sizes() {return vector_group_sizes;};
    int *get_adjacent_ids() {return adjacent_ids;};
    _TEdgeWeight *get_adjacent_weights() {return adjacent_weights;};

    void init_from_graph(long long *_csr_adjacent_ptrs, int *_csr_adjacent_ids, _TEdgeWeight *_csr_adjacent_weights,
                         int _first_vertex, int _last_vertex);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "vector_extension.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
