#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
#include "gpu/bfs_gpu.cuh"
#endif

#include <string>
#include "change_state/change_state.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


class BFS
{
private:
    GraphPrimitivesNEC graph_API;
    FrontierNEC frontier;

    #ifdef __USE_NEC_SX_AURORA__
    void nec_top_down_compute_step(UndirectedGraph &_graph, int *_levels,
                                   int _current_level, int &_vis, int &_in_lvl, bool _compute_stats);
    #endif

    #ifdef __USE_NEC_SX_AURORA__
    void nec_bottom_up_compute_step(UndirectedGraph &_graph, int *_levels, int *_connections_array,
                                    int _current_level, int &_vis, int &_in_lvl, bool _use_vector_extension);
    #endif
public:
    BFS(UndirectedGraph &_graph);
    ~BFS();
    
    void allocate_result_memory(int _vertices_count, int **_levels);
    void free_result_memory    (int *_levels);
    
    #ifdef __USE_GPU__
    void allocate_device_result_memory(int _vertices_count, int **_device_levels);
    void free_device_result_memory    (int *_device_levels);
    
    void copy_result_to_host(int *_host_levels, int *_device_levels, int _vertices_count);
    #endif

    #ifdef __USE_GPU__
    void gpu_top_down(UndirectedGraph &_graph, int *_levels, int _source_vertex);
    #endif

    #ifdef __USE_GPU__
    void gpu_direction_optimizing(UndirectedGraph &_graph, int *_levels, int _source_vertex);
    #endif

    #ifdef __USE_NEC_SX_AURORA__
    void nec_top_down(UndirectedGraph &_graph, int *_levels, int _source_vertex);
    #endif

    #ifdef __USE_NEC_SX_AURORA__
    double nec_direction_optimizing(UndirectedGraph &_graph, int *_levels,
                                  int _source_vertex);
    #endif

    void seq_top_down(UndirectedGraph &_graph, int *_levels, int _source_vertex);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "bfs.hpp"
#include "seq_bfs.hpp"
#include "gpu_bfs.hpp"
#include "nec_bfs.hpp"
#include "change_state/change_state.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

