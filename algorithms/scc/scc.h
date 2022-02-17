#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <stack>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define SCC StronglyConnectedComponents

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INIT_TREE 1
#define ERROR_IN_PIVOT -1
#define IS_NOT_ACTIVE 0
#define IS_ACTIVE 1

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// implements multiple SCC algorithms
class StronglyConnectedComponents
{
public:
    template <typename _T>
    static void trim_step(VGL_Graph &_graph,
                          VGL_GRAPH_ABSTRACTIONS &_graph_API,
                          VGL_FRONTIER &_frontier,
                          VerticesArray<_T> &_forward_result,
                          VerticesArray<_T> &_backward_result,
                          VerticesArray<_T> &_trees,
                          VerticesArray<_T> &_active);

    template <typename _T>
    static void process_result(VGL_Graph &_graph,
                               VGL_GRAPH_ABSTRACTIONS &_graph_API,
                               VGL_FRONTIER &_frontier,
                               VerticesArray<_T> &_forward_result,
                               VerticesArray<_T> &_backward_result,
                               VerticesArray<_T> &_trees,
                               VerticesArray<_T> &_active,
                               int _last_tree);

    template <typename _T>
    static void bfs_reach(VGL_Graph &_graph,
                          VGL_GRAPH_ABSTRACTIONS &_graph_API,
                          VGL_FRONTIER &_frontier,
                          VerticesArray<_T> &_bfs_result,
                          int _source_vertex,
                          TraversalDirection _traversal_direction);

    template <typename _T>
    static void FB_step(VGL_Graph &_graph,
                        VGL_GRAPH_ABSTRACTIONS &_graph_API,
                        VGL_FRONTIER &_frontier,
                        VerticesArray<_T> &_trees,
                        VerticesArray<_T> &_forward_result,
                        VerticesArray<_T> &_backward_result,
                        VerticesArray<_T> &_active,
                        int _processed_tree,
                        int &_last_tree);

    template <typename _T>
    static int select_pivot(VGL_Graph &_graph,
                     VGL_GRAPH_ABSTRACTIONS &_graph_API,
                     VGL_FRONTIER &_frontier,
                     VerticesArray<_T> &_trees,
                     int _tree_num);

    static void seq_tarjan_kernel(VGL_Graph &_graph,
                                  int _root,
                                  VerticesArray<int> &_disc,
                                  VerticesArray<int> &_low,
                                  stack<int> &_st,
                                  VerticesArray<bool> &_stack_member,
                                  VerticesArray<int> &_components);

    template <typename _T>
    static double vgl_forward_backward(VGL_Graph &_graph, VerticesArray<_T> &_components);

    static double seq_tarjan(VGL_Graph &_graph, VerticesArray<int> &_components);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "seq_scc.hpp"
#include "scc.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

