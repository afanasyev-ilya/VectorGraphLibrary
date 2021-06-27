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

class CSRGraph: public BaseGraph
{
private:
    long long     *vertex_pointers;
    int           *adjacent_ids;

    void alloc(int _vertices_count, long long _edges_count);
    void free();
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
    bool save_to_binary_file(string _file_name) {return false;};
    bool load_from_binary_file(string _file_name) {return false;};

    // resize graph
    void resize(int _vertices_count, long long _edges_count);

    /* import and preprocess API */
    // creates UndirectedVectCSRGraph format from EdgesListGraph
    void import(EdgesListGraph &_old_graph);

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

