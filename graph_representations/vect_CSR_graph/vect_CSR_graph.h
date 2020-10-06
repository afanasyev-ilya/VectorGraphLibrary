#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <vector>
#include <algorithm>
#include <string>
#include <fstream>
#include <stdio.h>

#include "../../common/cmd_parser/parser_options.h"
#include "../../common/memory_API/memory_API.h"
#include "../../graph_processing_API/framework_types.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class VectCSRGraph : public BaseGraph
{
private:
    ExtendedCSRGraph *outgoing_edges;
    ExtendedCSRGraph *incoming_edges;
public:
    VectCSRGraph(int _vertices_count = 1, long long _edges_count = 1);
    ~VectCSRGraph();

    void import_graph(EdgesListGraph &_copy_graph);

    ExtendedCSRGraph *get_outgoing_graph_ptr() {return outgoing_edges;};
    ExtendedCSRGraph *get_incoming_graph_ptr() {return incoming_edges;};

    ExtendedCSRGraph *get_direction_graph_ptr(TraversalDirection _direction);

    inline long long get_edges_count_in_outgoing_ve() {return outgoing_edges->get_edges_count_in_ve();};
    inline long long get_edges_count_in_incoming_ve() {return incoming_edges->get_edges_count_in_ve();};

    int renumber_to_vectCSR(int _id); // original -> vectCSR
    int renumber_to_original(int _id); // vectCSR -> original

    template <typename _T>
    void convert_to_inner_representation(_T* _per_vertex_data); // original -> vectCSR

    template <typename _T>
    void convert_from_inner_representation(_T* _per_vertex_data); // vectCSR -> original

    void print();
    //void print_with_weights(EdgesArrayNec<_TVertexValue, _TEdgeWeight, _TEdgeWeight> &_weights);

    void print_stats() {};
    void save_to_graphviz_file(string file_name, VisualisationMode _visualisation_mode = VISUALISE_AS_DIRECTED) {};
    bool save_to_binary_file(string file_name) {};
    bool load_from_binary_file(string file_name) {};
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "vect_CSR_graph.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

