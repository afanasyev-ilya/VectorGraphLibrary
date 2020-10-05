#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define MF MaxFlow<_TVertexValue, _TEdgeWeight>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
class MaxFlow
{
private:
    inline _TEdgeWeight get_flow(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int _src_id, int _dst_id);
    inline void subtract_flow(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int _src_id, int _dst_id, _TEdgeWeight update_val);
    inline void add_flow(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int _src_id, int _dst_id, _TEdgeWeight update_val);

    #ifdef __USE_NEC_SX_AURORA__
    bool nec_bfs(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int _source, int _sink,
                 int *_parents, int *_levels, GraphPrimitivesNEC &_graph_API, FrontierNEC<_TVertexValue, _TEdgeWeight> &_frontier);
    #endif

    bool seq_bfs(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int _source, int _sink, int *_parents);
public:
    MaxFlow() {};
    ~MaxFlow() {};

    #ifdef __USE_NEC_SX_AURORA__
    _TEdgeWeight nec_ford_fulkerson(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                    int _source, int _sink);
    #endif

    _TEdgeWeight seq_ford_fulkerson(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                    int _source, int _sink);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "mf.hpp"
#include "seq_mf.hpp"
#include "nec_mf.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

