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

#include "vg/vg.h"
#include "cell_c_vg/cell_c_vg.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class CSR_VG_Graph: public UndirectedGraph
{
private:
    long long     *vertex_pointers;
    int           *adjacent_ids;

    vgl_sort_indexes *edges_reorder_indexes; // allows to convert edges (and weights) from sorted to original order

    CSRVertexGroup vertex_groups[CSR_VERTEX_GROUPS_NUM];

    void create_vertex_groups();

    void alloc(int _vertices_count, long long _edges_count);
    void free();

    /* dummy reorder API */
    void reorder_to_sorted(char *_data, char *_buffer, size_t _elem_size) {return;};
    void reorder_to_original(char *_data, char *_buffer, size_t _elem_size) {return;};
    int reorder_to_sorted(int _vertex_id) { return _vertex_id; };
    int reorder_to_original(int _vertex_id) { return _vertex_id; };

    void construct_unsorted_csr(EdgesContainer &_edges_container);
    void copy_edges_indexes(vgl_sort_indexes *_sort_indexes);

    /* file load/store API */
    void save_main_content_to_binary_file(FILE *_graph_file) final;
    void load_main_content_from_binary_file(FILE *_graph_file) final;
public:
    CSRVertexGroupCellC cell_c_vertex_groups[CELL_C_VERTEX_GROUPS_NUM];

    CSR_VG_Graph(int _vertices_count = 1, long long _edges_count = 1);
    CSR_VG_Graph(const CSR_VG_Graph &_copy);
    ~CSR_VG_Graph();

    /* get API */
    inline long long *get_vertex_pointers() {return vertex_pointers;};
    inline int       *get_adjacent_ids()    {return adjacent_ids;};

    inline any_arch_func int get_connections_count(int _src_id) final {return vertex_pointers[_src_id+1] - vertex_pointers[_src_id];};
    inline any_arch_func int get_edge_dst(int _src_id, int _edge_pos) final {return adjacent_ids[vertex_pointers[_src_id] + _edge_pos];};

    inline size_t get_edges_array_index(int _v, int _edge_pos) final { return vertex_pointers[_v] + _edge_pos; };
    inline size_t get_edges_array_direction_shift_size() final { return this->edges_count; };

    /* print API */
    void print();
    void print_size();
    void print_stats() {};
    size_t get_size();

    // resize graph
    void resize(int _vertices_count, long long _edges_count);

    /* import and preprocess API */
    // creates VectorCSR_VG_Graph format from EdgesListGraph
    void import(EdgesContainer &_edges_container);

    void reorder_edges_gather(char *_src, char *_dst, size_t _elem_size) final;
    void reorder_edges_scatter(char *_src, char *_dst, size_t _elem_size) final;

    /* vertices API */
    int select_random_nz_vertex();

    /* GPU specific (copy) API */
    #ifdef __USE_GPU__
    void move_to_device() final;
    void move_to_host() final;
    #endif

    friend class GraphAbstractions;
    friend class FrontierCSR_VG;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define LOAD_CSR_GRAPH_DATA(input_graph)  \
int vertices_count            = input_graph.get_vertices_count(); \
long long int edges_count     = input_graph.get_edges_count   (); \
\
long long    *vertex_pointers = input_graph.get_vertex_pointers   ();\
int          *adjacent_ids    = input_graph.get_adjacent_ids    ();\

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "vg/vg.hpp"
#include "cell_c_vg/cell_c_vg.hpp"
#include "csr_vg_graph.hpp"
#include "import.hpp"
#include "print.hpp"
#include "reorder.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

