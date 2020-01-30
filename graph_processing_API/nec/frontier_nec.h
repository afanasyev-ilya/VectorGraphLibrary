#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define NEC_IN_FRONTIER_FLAG 1
#define NEC_NOT_IN_FRONTIER_FLAG 0

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class FrontierNEC
{
private:
    int *frontier_ids;
    int *frontier_flags;

    int max_frontier_size;
    int current_frontier_size;
public:
    FrontierNEC(int _vertices_count);
    ~FrontierNEC();

    void set_all_active();
    /*void split_sorted_frontier(const long long *_vertex_pointers, int &_grid_threshold_start, int &_grid_threshold_end,
                               int &_block_threshold_start, int &_block_threshold_end,
                               int &_warp_threshold_start, int &_warp_threshold_end,
                               int &_thread_threshold_start, int &_thread_threshold_end);*/

    /*template <typename _TVertexValue, typename _TEdgeWeight, typename Condition>
    void generate_frontier(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, Condition condition_op);*/

    int size() {return current_frontier_size;};

    friend class GraphPrimitivesNEC;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "frontier_nec.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
