#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <fstream>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class EdgesListGraph : public BaseGraph
{
private:
    int *src_ids;
    int *dst_ids;
    
    void alloc(int _vertices_count, long long _edges_count);
    void free();
public:
    EdgesListGraph(int _vertices_count = 1, long long _edges_count = 1);
    ~EdgesListGraph();

    /* get API */
    inline int *get_src_ids() {return src_ids;};
    inline int *get_dst_ids() {return dst_ids;};

    /* print API */
    void print();
    void print_in_csr_format();
    void print_size() {};
    size_t get_size() {return 0;};

    /* file load/store API */
    void save_to_graphviz_file(string file_name, VisualisationMode _visualisation_mode = VISUALISE_AS_DIRECTED);
    bool save_to_binary_file(string file_name);
    bool load_from_binary_file(string file_name);

    /* GPU specific (copy) API */
    #ifdef __USE_GPU__
    void move_to_device() {throw "not implemented yet";};
    void move_to_host() {throw "not implemented yet";};
    #endif

    /* Further - VectCSRGraph specific API : reorder, working with double-directions, etc.*/
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

    // 2D segmenting preprocessing (each segment fits into LLC cache)
    void preprocess_into_segmented();

    // CSR-based preprocessing (vertices are sorted based on src_ids)
    void preprocess_into_csr_based(int *_work_buffer = NULL, vgl_sort_indexes *_sort_buffer = NULL);
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
