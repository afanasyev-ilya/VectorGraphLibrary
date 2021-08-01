#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <fstream>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class EdgesListGraph : public UndirectedGraph
{
private:
    int *src_ids;
    int *dst_ids;

    void reorder_to_sorted(char *_data, char *_buffer, size_t _elem_size) {return;};
    void reorder_to_original(char *_data, char *_buffer, size_t _elem_size) {return;};
    
    void alloc(int _vertices_count, long long _edges_count);
    void free();
public:
    EdgesListGraph(int _vertices_count = 1, long long _edges_count = 1);
    EdgesListGraph(const EdgesListGraph &_copy_graph);
    ~EdgesListGraph();

    /* get API */
    inline int *get_src_ids() {return src_ids;};
    inline int *get_dst_ids() {return dst_ids;};
    int get_connections_count(int _vertex_id) { throw "Error: get_connections_count not implemented in Edges List Graph"; };
    int get_edge_dst(int _src_id, int _edge_pos) { throw "Error: get_edge_dst not implemented in Edges List Graph"; };

    /* print API */
    void print();
    void print_in_csr_format();
    template <typename _T>
    void print_in_csr_format(EdgesArray_EL<_T> &_weights);
    void print_size();
    size_t get_size();

    /* file load/store API */
    void save_to_graphviz_file(string file_name, VisualisationMode _visualisation_mode = VISUALISE_AS_DIRECTED);
    bool save_to_binary_file(string file_name);
    bool load_from_binary_file(string file_name);

    /* GPU specific (copy) API */
    #ifdef __USE_GPU__
    void move_to_device() {throw "not implemented yet";};
    void move_to_host() {throw "not implemented yet";};
    #endif

    /* Further - VGL_Graph specific API : reorder, working with double-directions, etc.*/
    // resize graph
    void resize(int _vertices_count, long long _edges_count);
    void clear() {free();};

    // transpose edges list graph (implemented as fast pointer swap)
    void transpose(); // TODO should it be in base Graph? basic graph operations API

    // allow to renumber vertices based on indexes provided in conversion array
    void renumber_vertices(int *_conversion_array, int *_work_buffer = NULL);

    /* import and preprocess API */
    // allows to import form 2 arrays (src_ids and dst_ids)
    void import(int *_src_ids, int *_dst_ids, int _vertices_count, long long _edges_count);
    void import(EdgesContainer &_edges_container) {cout << "HERE EL" << endl;};

    // 2D segmenting preprocessing (each segment fits into LLC cache)
    void preprocess_into_segmented();

    // CSR-based preprocessing (vertices are sorted based on src_ids)
    void preprocess_into_csr_based(int *_work_buffer = NULL, vgl_sort_indexes *_sort_buffer = NULL);

    /* remaining API */
    void operator = (const EdgesListGraph &_copy_graph);

    void remove_loops_and_multiple_arcs();

    int reorder_to_sorted(int _vertex_id) { return _vertex_id; };
    int reorder_to_original(int _vertex_id) { return _vertex_id; };
    int select_random_vertex() { return rand() % this->vertices_count; };
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "edges_list_graph.hpp"
#include "preprocess_into_segmented.hpp"
#include "preprocess_into_csr_based.hpp"
#include "print.hpp"
#include "import.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define LOAD_EDGES_LIST_GRAPH_DATA(input_graph)           \
int vertices_count    = input_graph.get_vertices_count(); \
int edges_count       = input_graph.get_edges_count   (); \
int *src_ids          = input_graph.get_src_ids       (); \
int *dst_ids          = input_graph.get_dst_ids       (); \

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
