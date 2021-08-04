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

class CSRGraph: public UndirectedGraph
{
private:
    long long     *vertex_pointers;
    int           *adjacent_ids;

    void alloc(int _vertices_count, long long _edges_count);
    void free();

    void construct_unsorted_csr(EdgesContainer &_edges_container);
public:
    CSRGraph(int _vertices_count = 1, long long _edges_count = 1);
    ~CSRGraph();

    inline int get_connections_count(int _src_id) final {return vertex_pointers[_src_id+1] - vertex_pointers[_src_id];};
    inline int get_edge_dst(int _src_id, int _edge_pos) final {return adjacent_ids[vertex_pointers[_src_id] + _edge_pos];};

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
    // creates VectorCSRGraph format from EdgesListGraph
    void import(EdgesContainer &_edges_container);

    friend class GraphAbstractions;
    friend class VGL_Graph;
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

