#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define NEC_IN_FRONTIER_FLAG 1
#define NEC_NOT_IN_FRONTIER_FLAG 0

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define FRONTIER_TYPE_CHANGE_THRESHOLD 0.2

enum FrontierType {
    ALL_ACTIVE_FRONTIER = 2,
    SPARSE_FRONTIER = 1,
    DENSE_FRONTIER = 0
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class FrontierNEC
{
private:
    int *frontier_flags;
    int *frontier_ids;
    int *work_buffer;

    int vector_engine_part_size;
    int vector_core_part_size;
    int collective_part_size;

    FrontierType frontier_type;

    int current_frontier_size;
    int max_frontier_size;

    template <typename _TVertexValue, typename _TEdgeWeight, typename Condition>
    void filter(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, Condition condition_op);
public:
    template <typename _TVertexValue, typename _TEdgeWeight>
    FrontierNEC(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph);
    FrontierNEC(int _vertices_count);
    ~FrontierNEC();

    int size() {return current_frontier_size;};
    FrontierType type() {return frontier_type;};

    void set_all_active();

    void print_frontier_info();

    friend class GraphPrimitivesNEC;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "frontier_nec.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
