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

    /* dummy reorder API */
    void reorder_to_sorted(char *_data, char *_buffer, size_t _elem_size) {return;};
    void reorder_to_original(char *_data, char *_buffer, size_t _elem_size) {return;};
    int reorder_to_sorted(int _vertex_id) { return _vertex_id; };
    int reorder_to_original(int _vertex_id) { return _vertex_id; };

    void construct_unsorted_csr(EdgesContainer &_edges_container);
public:
    CSRGraph(int _vertices_count = 1, long long _edges_count = 1);
    ~CSRGraph();

    inline int get_connections_count(int _src_id) final {return vertex_pointers[_src_id+1] - vertex_pointers[_src_id];};
    inline int get_edge_dst(int _src_id, int _edge_pos) final {return adjacent_ids[vertex_pointers[_src_id] + _edge_pos];};

    /* print API */
    void print();
    void print_size();
    void print_stats() {};

    size_t get_size();

    /* file load/store API */
    bool save_to_binary_file(string _file_name);
    bool load_from_binary_file(string _file_name);

    // resize graph
    void resize(int _vertices_count, long long _edges_count);

    /* import and preprocess API */
    // creates VectorCSRGraph format from EdgesListGraph
    void import(EdgesContainer &_edges_container);

    /* vertices API */
    int select_random_vertex() { return rand() % this->vertices_count; };

    friend class GraphAbstractions;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "csr_graph.hpp"
#include "import.hpp"
#include "print.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

