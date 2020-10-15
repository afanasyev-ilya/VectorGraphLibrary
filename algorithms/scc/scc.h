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
    #ifdef __USE_NEC_SX_AURORA__
    template <typename _T>
    static void trim_step(VectCSRGraph &_graph,
                          GraphAbstractionsNEC &_graph_API,
                          FrontierNEC &_frontier,
                          VerticesArrayNEC<_T> &_forward_result,
                          VerticesArrayNEC<_T> &_backward_result,
                          VerticesArrayNEC<_T> &_trees,
                          VerticesArrayNEC<_T> &_active);
    #endif

    #ifdef __USE_NEC_SX_AURORA__
    template <typename _T>
    static void process_result(VectCSRGraph &_graph,
                               GraphAbstractionsNEC &_graph_API,
                               FrontierNEC &_frontier,
                               VerticesArrayNEC<_T> &_forward_result,
                               VerticesArrayNEC<_T> &_backward_result,
                               VerticesArrayNEC<_T> &_trees,
                               VerticesArrayNEC<_T> &_active,
                               int _last_tree);
    #endif

    #ifdef __USE_NEC_SX_AURORA__
    template <typename _T>
    static void bfs_reach(VectCSRGraph &_graph,
                          GraphAbstractionsNEC &_graph_API,
                          FrontierNEC &_frontier,
                          VerticesArrayNEC<_T> &_bfs_result,
                          int _source_vertex,
                          TraversalDirection _traversal_direction);
    #endif

    #ifdef __USE_NEC_SX_AURORA__
    template <typename _T>
    static void FB_step(VectCSRGraph &_graph,
                        GraphAbstractionsNEC &_graph_API,
                        FrontierNEC &_frontier,
                        VerticesArrayNEC<_T> &_trees,
                        VerticesArrayNEC<_T> &_forward_result,
                        VerticesArrayNEC<_T> &_backward_result,
                        VerticesArrayNEC<_T> &_active,
                        int _processed_tree,
                        int &_last_tree);
    #endif

    #ifdef __USE_NEC_SX_AURORA__
    template <typename _T>
    static int select_pivot(VectCSRGraph &_graph,
                     GraphAbstractionsNEC &_graph_API,
                     FrontierNEC &_frontier,
                     VerticesArrayNEC<_T> &_trees,
                     int _tree_num);
    #endif

    static void seq_tarjan_kernel(VectCSRGraph &_graph,
                                  int _root,
                                  VerticesArrayNEC<int> &_disc,
                                  VerticesArrayNEC<int> &_low,
                                  stack<int> &_st,
                                  VerticesArrayNEC<bool> &_stack_member,
                                  VerticesArrayNEC<int> &_components);
public:
    #ifdef __USE_NEC_SX_AURORA__
    template <typename _T>
    static void nec_forward_backward(VectCSRGraph &_graph, VerticesArrayNEC<_T> &_components);
    #endif

    static void seq_tarjan(VectCSRGraph &_graph, VerticesArrayNEC<int> &_components);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "seq_scc.hpp"
#include "nec_scc.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

