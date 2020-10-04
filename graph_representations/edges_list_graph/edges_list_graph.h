#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <fstream>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
class EdgesListGraph : public BaseGraph<_TVertexValue, _TEdgeWeight>
{
private:
    int *src_ids;
    int *dst_ids;
    _TEdgeWeight *weights;
    
    void alloc(int _vertices_count, long long _edges_count);
    void free();
public:
    EdgesListGraph(int _vertices_count = 1, long long _edges_count = 1);
    ~EdgesListGraph();
    
    inline int *get_src_ids() {return src_ids;};
    inline int *get_dst_ids() {return dst_ids;};
    inline _TEdgeWeight *get_weights() {return weights;};
    
    void resize(int _vertices_count, long long _edges_count);

    void transpose();
    
    void print();
    void print_in_csr_format();
    void print_stats();
    
    void save_to_graphviz_file(string file_name, VisualisationMode _visualisation_mode = VISUALISE_AS_DIRECTED);
    bool save_to_binary_file(string file_name);
    bool load_from_binary_file(string file_name);

    // allow to renumber vertices based on indexes provided in conversion array
    void renumber_vertices(int *_conversion_array, int *_work_buffer = NULL);

    void preprocess_into_segmented();

    #ifdef __USE_ASL__
    void preprocess_into_csr_based(int *_work_buffer = NULL, asl_int_t *_asl_buffer = NULL);
    #endif
    
    #ifdef __USE_GPU__
    void move_to_device() {throw "not implemented yet";};
    void move_to_host() {throw "not implemented yet";};
    #endif
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "edges_list_graph.hpp"
#include "preprocess_into_segmented.hpp"
#include "preprocess_into_csr_based.hpp"
#include "print.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define LOAD_EDGES_LIST_GRAPH_DATA(input_graph)           \
int vertices_count    = input_graph.get_vertices_count(); \
int edges_count       = input_graph.get_edges_count   (); \
int *src_ids          = input_graph.get_src_ids       (); \
int *dst_ids          = input_graph.get_dst_ids       (); \
_TEdgeWeight *weights = input_graph.get_weights       (); \

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
