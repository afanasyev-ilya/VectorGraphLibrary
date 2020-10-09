#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <vector>
#include <algorithm>
#include <string>
#include <fstream>
#include <stdio.h>

#include "../../common/cmd_parser/parser_options.h"
#include "../../common/memory_API/memory_API.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class VectCSRGraph : public BaseGraph
{
private:
    ExtendedCSRGraph *outgoing_graph;
    ExtendedCSRGraph *incoming_graph;

    long long *edges_reorder_buffer;

    void resize_edges_reorder_buffer();
public:
    VectCSRGraph(int _vertices_count = 1, long long _edges_count = 1);
    ~VectCSRGraph();

    // initializes VectCSR graph from EdgesList graph
    void import_graph(EdgesListGraph &_copy_graph);

    // get pointers to the specific undirected part of graph (incoming or outgoing ids)
    ExtendedCSRGraph *get_outgoing_graph_ptr() {return outgoing_graph;};
    ExtendedCSRGraph *get_incoming_graph_ptr() {return incoming_graph;};
    ExtendedCSRGraph *get_direction_graph_ptr(TraversalDirection _direction);

    // allows to get vector engine size
    inline long long get_edges_count_in_outgoing_ve() {return outgoing_graph->get_edges_count_in_ve();};
    inline long long get_edges_count_in_incoming_ve() {return incoming_graph->get_edges_count_in_ve();};

    // allows to reorder a single vertex ID
    int reorder(int _vertex_id, TraversalDirection _input_dir, TraversalDirection _output_dir);

    // allows to reorder verticesArrayw
    template <typename _T>
    void reorder_to_original(VerticesArrayNec<_T> &_data);
    template <typename _T>
    void reorder_to_scatter(VerticesArrayNec<_T> &_data);
    template <typename _T>
    void reorder_to_gather(VerticesArrayNec<_T> &_data);
    template <typename _T>
    void reorder(VerticesArrayNec<_T> &_data, TraversalDirection _output_dir);

    // allows to reorder edges array to secondary direction (gather)
    template <typename _T>
    void reorder_edges_to_gather(_T *_incoming_csr_ptr, _T *_outgoing_csr_ptr); // TODO name fix?

    // prints graph if it is small enough
    void print();

    // prints graph with weights attached if it is small enough
    template <typename _T>
    void print_with_weights(EdgesArrayNec<_T> &_weights);

    // file load/store API
    void save_to_graphviz_file(string file_name, VisualisationMode _visualisation_mode = VISUALISE_AS_DIRECTED) {};
    bool save_to_binary_file(string file_name) {};
    bool load_from_binary_file(string file_name) {};

    // selects random vertex with non-zero outgoing and incoming degree
    int select_random_vertex(TraversalDirection _direction = ORIGINAL);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "vect_CSR_graph.hpp"
#include "reorder.hpp"
#include "print.hpp"
#include "preprocess.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

