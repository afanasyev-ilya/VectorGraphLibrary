#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <vector>
#include <algorithm>
#include <string>
#include <fstream>
#include <stdio.h>

#include "../../common/cmd_parser/parser_options.h"
#include "../common/tmp_edge_data.h"
#include "../../common/memory_API/memory_API.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
class VectCSRGraph : public BaseGraph<_TVertexValue, _TEdgeWeight>
{
private:
    ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> *outgoing_edges;
    ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> *incoming_edges;

    int *out_to_inc_conversion;
    int *in_to_out_conversion;
public:
    VectCSRGraph(int _vertices_count = 1, long long _edges_count = 1);
    ~VectCSRGraph();

    void import_graph(EdgesListGraph<_TVertexValue, _TEdgeWeight> &_copy_graph);

    void print() {};
    void print_stats() {};
    void save_to_graphviz_file(string file_name, VisualisationMode _visualisation_mode = VISUALISE_AS_DIRECTED) {};
    bool save_to_binary_file(string file_name) {};
    bool load_from_binary_file(string file_name) {};
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "vect_CSR_graph.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

