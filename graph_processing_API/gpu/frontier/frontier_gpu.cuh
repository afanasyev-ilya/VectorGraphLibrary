#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../../framework_types.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define GPU_VWP_16_THRESHOLD_VALUE 16
#define GPU_VWP_8_THRESHOLD_VALUE 8
#define GPU_VWP_4_THRESHOLD_VALUE 4
#define GPU_VWP_2_THRESHOLD_VALUE 2

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class FrontierGPU
{
private:
    int *ids;
    int *flags;

    FrontierType type;

    int max_size;
    int current_size;

    void split_sorted_frontier(const long long *_vertex_pointers, int &_grid_threshold_start, int &_grid_threshold_end,
                               int &_block_threshold_start, int &_block_threshold_end,
                               int &_warp_threshold_start, int &_warp_threshold_end,
                               int &_vwp_16_threshold_start, int &_vwp_16_threshold_end,
                               int &_vwp_8_threshold_start, int &_vwp_8_threshold_end,
                               int &_vwp_4_threshold_start, int &_vwp_4_threshold_end,
                               int &_vwp_2_threshold_start, int &_vwp_2_threshold_end,
                               int &_thread_threshold_start, int &_thread_threshold_end);
public:
    FrontierGPU(int _vertices_count);

    void set_all_active();

    ~FrontierGPU();

    int size() {return current_size;};

    FrontierType get_type() {return type;};

    friend class GraphPrimitivesGPU;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "frontier_gpu.cu"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////