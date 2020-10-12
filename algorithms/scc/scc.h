#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <stack>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define SCC StronglyConnectedComponents

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INIT_COMPONENT -1
#define INIT_TREE 1

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class StronglyConnectedComponents
{
private:
    #ifdef __USE_NEC_SX_AURORA__
    template <typename _T>
    static void bfs_reach(VectCSRGraph &_graph,
                  GraphAbstractionsNEC &_graph_API,
                  FrontierNEC &_frontier,
                  VerticesArrayNec<_T> &_bfs_result,
                  int _source_vertex,
                  TraversalDirection _traversal_direction);
    #endif

    #ifdef __USE_NEC_SX_AURORA__
    template <typename _T>
    static void FB_step(VectCSRGraph &_graph, GraphAbstractionsNEC &_graph_API, FrontierNEC &_frontier,
                        VerticesArrayNec<_T> &_components, VerticesArrayNec<_T> &_trees,
                        VerticesArrayNec<_T> &_forward_result, VerticesArrayNec<_T> &_backward_result, int _tree_num);
    #endif

    #ifdef __USE_NEC_SX_AURORA__
    template <typename _T>
    static int select_pivot(VectCSRGraph &_graph,
                     GraphAbstractionsNEC &_graph_API,
                     FrontierNEC &_frontier,
                     VerticesArrayNec<_T> &_trees,
                     int _tree_num);
    #endif

    static void seq_tarjan_kernel(VectCSRGraph &_graph,
                                  int _root,
                                  VerticesArrayNec<int> &_disc,
                                  VerticesArrayNec<int> &_low,
                                  stack<int> &_st,
                                  VerticesArrayNec<bool> &_stack_member,
                                  VerticesArrayNec<int> &_components);
public:
    #ifdef __USE_NEC_SX_AURORA__
    template <typename _T>
    static void nec_forward_backward(VectCSRGraph &_graph, VerticesArrayNec<_T> &_components);
    #endif

    static void seq_tarjan(VectCSRGraph &_graph, VerticesArrayNec<int> &_components);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "seq_scc.hpp"
#include "nec_scc.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

