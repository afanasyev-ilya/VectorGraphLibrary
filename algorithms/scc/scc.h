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

class StronglyConnectedComponents
{
private:
    #if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
    template <typename _T>
    static void trim_step(VectCSRGraph &_graph,
                          VGL_GRAPH_ABSTRACTIONS &_graph_API,
                          VGL_FRONTIER &_frontier,
                          VerticesArray<_T> &_forward_result,
                          VerticesArray<_T> &_backward_result,
                          VerticesArray<_T> &_trees,
                          VerticesArray<_T> &_active);
    #endif

    #if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
    template <typename _T>
    static void process_result(VectCSRGraph &_graph,
                               VGL_GRAPH_ABSTRACTIONS &_graph_API,
                               VGL_FRONTIER &_frontier,
                               VerticesArray<_T> &_forward_result,
                               VerticesArray<_T> &_backward_result,
                               VerticesArray<_T> &_trees,
                               VerticesArray<_T> &_active,
                               int _last_tree);
    #endif

    #if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
    template <typename _T>
    static void bfs_reach(VectCSRGraph &_graph,
                          VGL_GRAPH_ABSTRACTIONS &_graph_API,
                          VGL_FRONTIER &_frontier,
                          VerticesArray<_T> &_bfs_result,
                          int _source_vertex,
                          TraversalDirection _traversal_direction);
    #endif

    #if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
    template <typename _T>
    static void FB_step(VectCSRGraph &_graph,
                        VGL_GRAPH_ABSTRACTIONS &_graph_API,
                        VGL_FRONTIER &_frontier,
                        VerticesArray<_T> &_trees,
                        VerticesArray<_T> &_forward_result,
                        VerticesArray<_T> &_backward_result,
                        VerticesArray<_T> &_active,
                        int _processed_tree,
                        int &_last_tree);
    #endif

    #if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
    template <typename _T>
    static int select_pivot(VectCSRGraph &_graph,
                     VGL_GRAPH_ABSTRACTIONS &_graph_API,
                     VGL_FRONTIER &_frontier,
                     VerticesArray<_T> &_trees,
                     int _tree_num);
    #endif

    static void seq_tarjan_kernel(VectCSRGraph &_graph,
                                  int _root,
                                  VerticesArray<int> &_disc,
                                  VerticesArray<int> &_low,
                                  stack<int> &_st,
                                  VerticesArray<bool> &_stack_member,
                                  VerticesArray<int> &_components);
public:
    #if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
    template <typename _T>
    static void nec_forward_backward(VectCSRGraph &_graph, VerticesArray<_T> &_components);
    #endif

    #ifdef __USE_GPU__
    template <typename _T>
    static void gpu_forward_backward(VectCSRGraph &_graph, VerticesArray<_T> &_components);
    #endif

    static void seq_tarjan(VectCSRGraph &_graph, VerticesArray<int> &_components);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "seq_scc.hpp"
#include "scc.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

