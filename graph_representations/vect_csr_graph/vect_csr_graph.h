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
    UndirectedCSRGraph *outgoing_graph;
    UndirectedCSRGraph *incoming_graph;

    long long *vertices_reorder_buffer;
    long long *edges_reorder_indexes;

    void resize_helper_arrays();

    template <typename _T>
    bool vertices_buffer_can_be_used(VerticesArray<_T> &_data);
    template <typename _T>
    void reorder_to_original(VerticesArray<_T> &_data);
    template <typename _T>
    void reorder_to_scatter(VerticesArray<_T> &_data);
    template <typename _T>
    void reorder_to_gather(VerticesArray<_T> &_data);
public:
    VectCSRGraph(int _vertices_count = 1, long long _edges_count = 1);
    ~VectCSRGraph();

    /* print API */
    void print();
    void print_size();
    size_t get_size();
    template <typename _T>
    void print_with_weights(EdgesArray<_T> &_weights);

    /* file load/store API */
    bool save_to_binary_file(string file_name) {};
    bool load_from_binary_file(string file_name) {};

    /* GPU specific (copy) API */
    #ifdef __USE_GPU__
    virtual void move_to_device() = 0;
    virtual void move_to_host() = 0;
    #endif

    /* Further - VectCSRGraph specific API : reorder, working with double-directions, etc.*/

    // get pointers to the specific undirected part of graph (incoming or outgoing ids)
    UndirectedCSRGraph *get_outgoing_graph_ptr() {return outgoing_graph;};
    UndirectedCSRGraph *get_incoming_graph_ptr() {return incoming_graph;};
    UndirectedCSRGraph *get_direction_graph_ptr(TraversalDirection _direction);

    // allows to get vector engine size
    inline long long get_edges_count_in_outgoing_ve() {return outgoing_graph->get_edges_count_in_ve();};
    inline long long get_edges_count_in_incoming_ve() {return incoming_graph->get_edges_count_in_ve();};

    /* reorder API */
    // allows to reorder a single vertex ID in arbitrary direction
    int reorder(int _vertex_id, TraversalDirection _input_dir, TraversalDirection _output_dir);

    // allows to reorder verticesArray in arbitrary direction
    template <typename _T>
    void reorder(VerticesArray<_T> &_data, TraversalDirection _output_dir);

    // allows to reorder frontier in arbitrary direction
    void reorder(FrontierNEC &_data, TraversalDirection _output_dir);

    // allows to reorder edges array to secondary direction (gather)
    template <typename _T>
    void reorder_edges_to_gather(_T *_incoming_csr_ptr, _T *_outgoing_csr_ptr); // TODO name fix?

    // selects random vertex with non-zero outgoing and incoming degree
    int select_random_vertex(TraversalDirection _direction = ORIGINAL);

    // performs simple graph visualization using GraphViz API
    template <typename _TVertexValue>
    void save_to_graphviz_file(string _file_name, VerticesArray<_TVertexValue> &_vertex_data);

    /* import and preprocess API */
    // creates VectCSRGraph format from EdgesListGraph
    void import(EdgesListGraph &_copy_graph);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "vect_csr_graph.hpp"
#include "reorder.hpp"
#include "print.hpp"
#include "import.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

