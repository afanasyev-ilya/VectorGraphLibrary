#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
#include "gpu/traditional_bfs.h"
#endif

#include <string>
#include "change_state/change_state.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
class BFS
{
private:
    GraphPrimitivesNEC graph_API;
    FrontierNEC frontier;

    #ifdef __USE_NEC_SX_AURORA__
    void nec_top_down_compute_step(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int *_levels,
                                   int _current_level, int &_vis, int &_in_lvl);
    #endif

    #ifdef __USE_NEC_SX_AURORA__
    void nec_bottom_up_compute_step(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int *_levels,
                                    int _current_level, int &_vis, int &_in_lvl, bool _use_vector_extension);
    #endif

    void performance_stats(string _name, double _time, long long _edges_count, int _iterations_count);
public:
    BFS(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph);
    ~BFS();
    
    void allocate_result_memory(int _vertices_count, int **_levels);
    void free_result_memory    (int *_levels);
    
    #ifdef __USE_GPU__
    void allocate_device_result_memory(int _vertices_count, int **_device_levels);
    void free_device_result_memory    (int *_device_levels);
    
    void copy_result_to_host(int *_host_levels, int *_device_levels, int _vertices_count);
    #endif

    #ifdef __USE_GPU__
    void gpu_direction_optimising_BFS(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int *_levels, int _source_vertex);
    #endif

    #ifdef __USE_NEC_SX_AURORA__
    void nec_top_down(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int *_levels, int _source_vertex);
    #endif

    #ifdef __USE_NEC_SX_AURORA__
    void nec_bottom_up(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int *_levels, int _source_vertex);
    #endif

    #ifdef __USE_NEC_SX_AURORA__
    void nec_direction_optimising(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int *_levels,
                                  int _source_vertex);
    #endif

    void seq_top_down(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int *_levels, int _source_vertex);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "bfs.hpp"
#include "seq_bfs.hpp"
#include "gpu_bfs.hpp"
#include "nec_bfs.hpp"
#include "change_state/change_state.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

