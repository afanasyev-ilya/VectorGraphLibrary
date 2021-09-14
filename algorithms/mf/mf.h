#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define MF MaxFlow

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


class MaxFlow
{
private:
    template <typename _T>
    static inline _T get_flow(VGL_Graph &_graph, EdgesArray<_T> &_weights, int _src_id, int _dst_id);
    template <typename _T>
    static inline void subtract_flow(VGL_Graph &_graph, EdgesArray<_T> &_weights, int _src_id, int _dst_id, _T update_val);
    template <typename _T>
    static inline void add_flow(VGL_Graph &_graph, EdgesArray<_T> &_weights, int _src_id, int _dst_id, _T update_val);

    template <typename _T>
    static bool mf_bfs(VGL_Graph &_graph,
                       EdgesArray<_T> &_weights,
                       int _source,
                       int _sink,
                       VerticesArray<int> &_parents,
                       VerticesArray<int> &_levels,
                       VGL_GRAPH_ABSTRACTIONS &_graph_API,
                       VGL_FRONTIER &_frontier);

    template <typename _T>
    static bool seq_bfs(VGL_Graph &_graph, EdgesArray<_T> &_weights, int _source, int _sink, int *_parents);
public:
    template <typename _T>
    static double vgl_ford_fulkerson(VGL_Graph &_graph, EdgesArray<_T> &_flows, int _source, int _sink, _T &_max_flow);

    template <typename _T>
    static double seq_ford_fulkerson(VGL_Graph &_graph, EdgesArray<_T> &_flows, int _source, int _sink, _T &_max_flow);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "mf.hpp"
#include "seq_mf.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

