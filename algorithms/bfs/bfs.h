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
    // temporary buffers
    int *active_ids;
    int *active_vertices_buffer;
    
    // nec functions
    int nec_remove_zero_nodes(long long *_outgoing_ptrs, int _vertices_count, int *_levels);
    int nec_mark_zero_nodes(int _vertices_count, int *_levels);
    
    void nec_top_down_step(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, long long *_outgoing_ptrs,
                           int *_outgoing_ids, int _vertices_count, int _active_count,
                           int *_levels, int *_cached_levels, int _cur_level, int &_vis,
                           int &_in_lvl, int _threads_count);
    
    void nec_bottom_up_step(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, long long *_outgoing_ptrs,
                            int *_outgoing_ids, int _vertices_count, int _active_count, int *_levels,
                            int *_cached_levels, int _cur_level, int &_vis, int &_in_lvl,
                            int _threads_count, int *_vectorised_outgoing_ids, bool _use_vect_CSR_extension, int _non_zero_vertices_count,
                            double &_t_first, double &_t_second, double &_t_third);

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
    
    void nec_direction_optimising_BFS(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int *_levels, int _source_vertex);
    
    #ifdef __USE_GPU__
    void gpu_direction_optimising_BFS(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int *_levels, int _source_vertex);
    #endif

    #ifdef __USE_NEC_SX_AURORA__
    void nec_top_down(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int *_levels, int _source_vertex);
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

