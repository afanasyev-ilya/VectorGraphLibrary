#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class GraphPrimitivesNEC
{
private:
    //void split_frontier(FrontierGPU &_frontier) {}
public:
    GraphPrimitivesNEC() {};

    ~GraphPrimitivesNEC() {};

    //template <typename InitOperation>
    //void init(int size, InitOperation init_op);

    template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation>
    void advance(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                 int large_threshold_vertex,
                 int medium_threshold_vertex,
                 EdgeOperation edge_op);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_primitives_nec.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
