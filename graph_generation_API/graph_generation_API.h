//
//  graph_generation_API.h
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 14/04/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef graph_generation_API_h
#define graph_generation_API_h

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <math.h>
#include <sstream>
#include <string>

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

template <typename _TVertexValue, typename _TEdgeWeight>
class GraphGenerationAPI
{
public:
    static void random_uniform(EdgesListGraph<_TVertexValue, _TEdgeWeight> &_graph,
                               int _vertices_count,
                               long long _edges_count,
                               DirectionType _direction_type = DIRECTED_GRAPH);
    
    static void R_MAT(EdgesListGraph<_TVertexValue, _TEdgeWeight> &_graph,
                      int _vertices_count,
                      long long _edges_count,
                      int _a_prob,
                      int _b_prob,
                      int _c_prob,
                      int _d_prob,
                      DirectionType _direction_type = DIRECTED_GRAPH);
    
    static void SSCA2(EdgesListGraph<_TVertexValue, _TEdgeWeight> &_graph,
                      int _vertices_count,
                      int _max_clique_size);
    
    static void SCC_uniform(EdgesListGraph<_TVertexValue, _TEdgeWeight> &_graph,
                            int _vertices_count,
                            int _min_scc_size,
                            int _max_scc_size);
    
    static void init_from_txt_file(EdgesListGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                   string _txt_file_name,
                                   bool _append_with_reverse_edges = true);
};


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_generation_API.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* graph_generation_API_h */
