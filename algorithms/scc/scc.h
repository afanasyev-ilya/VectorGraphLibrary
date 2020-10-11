#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <stack>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define SCC StronglyConnectedComponents

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class StronglyConnectedComponents
{
private:
    #ifdef __USE_NEC_SX_AURORA__
    void bfs_reach(VectCSRGraph &_graph,
                  GraphAbstractionsNEC &_graph_API,
                  FrontierNEC &frontier,
                  VerticesArrayNec<int> &_bfs_result,
                  int _source_vertex);
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
    static void nec_forward_backward(VectCSRGraph &_graph, VerticesArrayNec<int> &_components);
    #endif

    static void seq_tarjan(VectCSRGraph &_graph, VerticesArrayNec<int> &_components);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "seq_scc.hpp"
#include "nec_scc.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

