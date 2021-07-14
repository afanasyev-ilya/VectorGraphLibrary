#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <vector>
#include <algorithm>
#include <string>
#include <fstream>
#include <stdio.h>
#include <string.h>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
class VerticesArray;

template <typename _T>
class EdgesArray;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct CSRVertexGroup
{
    int *ids;
    int size;
    long long neighbours;

    CSRVertexGroup()
    {
        size = 1;
        neighbours = 0;
        MemoryAPI::allocate_array(&ids, size);
    }

    void resize(int _new_size)
    {
        size = _new_size;
        MemoryAPI::free_array(ids);
        MemoryAPI::allocate_array(&ids, size);
    }

    ~CSRVertexGroup()
    {
        MemoryAPI::free_array(ids);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class CSRGraph: public BaseGraph
{
private:
    long long     *vertex_pointers;
    int           *adjacent_ids;

    void alloc(int _vertices_count, long long _edges_count);
    void free();

    void construct_unsorted_csr(EdgesListGraph &_el_graph, bool _random_shuffle_required);

    void create_vertices_group_array(CSRVertexGroup &_group_data, int _bottom, int _top);

    int *data;
    int *result;
    void test_advance_changed_vl(CSRVertexGroup &_group_data, string _name);
    void test_advance_fixed_vl(CSRVertexGroup &_group_data, string _name);
    void test_advance_sparse(CSRVertexGroup &_group_data, string _name);
    void test_advance_virtual_warp(CSRVertexGroup &_group_data, string _name);
    void test_advance_sparse_packed(CSRVertexGroup &_group_data, string _name);
public:
    CSRGraph(int _vertices_count = 1, long long _edges_count = 1);
    ~CSRGraph();

    inline int get_connections_count(int _vertex_id) {return 0;};
    inline int get_edge_dst(int _src_id, int _edge_pos) {return 0;};

    /* print API */
    void print() {};
    void print_size() {};
    void print_stats() {};

    size_t get_size() {return 0;};

    /* file load/store API */
    bool save_to_binary_file(string _file_name);
    bool load_from_binary_file(string _file_name);

    // resize graph
    void resize(int _vertices_count, long long _edges_count);

    void test_advance();
    void test_full_advance();

    /* import and preprocess API */
    // creates UndirectedVectCSRGraph format from EdgesListGraph
    void import(EdgesListGraph &_old_graph, bool _random_shuffle_required = true);

    friend class GraphAbstractions;
    friend class VectCSRGraph;
    friend class ShardedCSRGraph;
    friend class EdgesListGraph;
    template<class _T> friend class EdgesArray_Vect;
    template<class _T> friend class EdgesArray_Sharded;
    template<class _T> friend class EdgesArray_EL;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "csr_graph.hpp"
#include "import.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

