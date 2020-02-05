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
    int *work_buffer;

    int max_frontier_size;
    int current_frontier_size;

    int shared_tmp_val_int;
    int get_non_zero_value(int _private_value);
public:
    FrontierNEC(int _vertices_count);
    ~FrontierNEC();

    template <typename Condition>
    void set_frontier_flags(Condition condition_op);

    void set_all_active();
    void split_sorted_frontier(const long long *_vertex_pointers, int &_large_threshold_start, int &_large_threshold_end,
                               int &_medium_threshold_start, int &_medium_threshold_end,
                               int &_small_threshold_start, int &_small_threshold_end);

    template <typename _TVertexValue, typename _TEdgeWeight, typename Condition>
    void generate_frontier(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, Condition condition_op);

    int size() {return current_frontier_size;};

    friend class GraphPrimitivesNEC;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "frontier_nec.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
