#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define GPU_IN_FRONTIER_FLAG 1
#define GPU_NOT_IN_FRONTIER_FLAG 0

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class FrontierGPU
{
private:
    int *frontier_ids;
    char *frontier_flags;
    int max_frontier_size;
    int current_frontier_size;
public:
    FrontierGPU(int _vertices_count);

    void set_all_active_frontier();
    void split_sorted_frontier(const long long *_vertex_pointers, int &_grid_threshold_start, int &_grid_threshold_end,
                               int &_block_threshold_start, int &_block_threshold_end,
                               int &_warp_threshold_start, int &_warp_threshold_end,
                               int &_thread_threshold_start, int &_thread_threshold_end);

    template <typename _TVertexValue, typename _TEdgeWeight, typename Condition>
    void generate_frontier(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, Condition condition_op);

    ~FrontierGPU();

    int size() {return current_frontier_size;};

    friend class GraphPrimitivesGPU;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "frontier.cu"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////