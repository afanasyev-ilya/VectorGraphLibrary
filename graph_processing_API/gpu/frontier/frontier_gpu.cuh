#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define GPU_VWP_16_THRESHOLD_VALUE 16
#define GPU_VWP_8_THRESHOLD_VALUE 8
#define GPU_VWP_4_THRESHOLD_VALUE 4
#define GPU_VWP_2_THRESHOLD_VALUE 2

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class FrontierGPU : public Frontier
{
private:
    // this is how NEC frontier is represented
    int *ids;
    int *flags;

    void init();

    void split_sorted_frontier(const long long *_vertex_pointers,
                               int &_block_threshold_start, int &_block_threshold_end,
                               int &_warp_threshold_start, int &_warp_threshold_end,
                               int &_vwp_16_threshold_start, int &_vwp_16_threshold_end,
                               int &_vwp_8_threshold_start, int &_vwp_8_threshold_end,
                               int &_vwp_4_threshold_start, int &_vwp_4_threshold_end,
                               int &_vwp_2_threshold_start, int &_vwp_2_threshold_end,
                               int &_thread_threshold_start, int &_thread_threshold_end);
public:
    /* constructors and destructors */
    FrontierGPU(VectCSRGraph &_graph, TraversalDirection _direction = SCATTER);
    ~FrontierGPU();

    /* Get API */
    int *get_flags() {return flags;};
    int *get_ids() {return ids;};

    /* Print API */
    void print_stats();
    void print();

    /* frontier modification API */
    inline void set_all_active();
    inline void add_vertex(int src_id);
    inline void add_group_of_vertices(int *_vertex_ids, int _number_of_vertices);

    friend class GraphAbstractionsGPU;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "frontier_gpu.cu"
#include "modification.cu"
#include "print.cu"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////