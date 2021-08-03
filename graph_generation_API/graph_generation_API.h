#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <math.h>
#include <sstream>
#include <string>
#include "edges_container.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum ConvertDirectionType
{
    DirectedToDirected = 0,
    DirectedToUndirected = 1,
    UndirectedToDirected = 2,
    UndirectedToUndirected = 3
};

enum DirectionType
{
    UNDIRECTED_GRAPH = 0,
    DIRECTED_GRAPH = 1
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class GraphGenerationAPI
{
public:
    static void random_uniform(EdgesContainer &_edges_container,
                               int _vertices_count,
                               long long _edges_count,
                               DirectionType _direction_type = DIRECTED_GRAPH);
    
    static void R_MAT(EdgesContainer &_edges_container,
                      int _vertices_count,
                      long long _edges_count,
                      int _a_prob,
                      int _b_prob,
                      int _c_prob,
                      int _d_prob,
                      DirectionType _direction_type = DIRECTED_GRAPH);
    
    /*static void SSCA2(EdgesListGraph &_graph,
                      int _vertices_count,
                      int _max_clique_size);
    
    static void SCC_uniform(EdgesListGraph &_graph,
                            int _vertices_count,
                            int _min_scc_size,
                            int _max_scc_size);
    
    static void init_from_txt_file(EdgesListGraph &_graph,
                                   string _txt_file_name,
                                   DirectionType _direction_type = DIRECTED_GRAPH);*/
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_generation_API.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
