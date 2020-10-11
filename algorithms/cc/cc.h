#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_INTEL__
#include <limits>
#include <cfloat>
#endif

#ifdef __USE_GPU__
#include "gpu/shiloach_vishkin_gpu.cuh"
#endif

#define COMPONENT_UNSET -1
#define FIRST_COMPONENT 1
#define SINGLE_VERTEX_COMPONENT -2
#define DUO_VERTEX_COMPONENT -3

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define CC ConnectedComponents

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


class ConnectedComponents
{
private:
    GraphPrimitivesNEC graph_API;
    FrontierNEC frontier;
    FrontierNEC bfs_frontier;

    double performance;
public:
    ConnectedComponents(UndirectedCSRGraph &_graph): frontier(_graph.get_vertices_count()), bfs_frontier(_graph.get_vertices_count()){};
    ~ConnectedComponents() {};

    void allocate_result_memory(int _vertices_count, int **_components);
    void free_result_memory    (int *_components);

    #ifdef __USE_NEC_SX_AURORA__
    void nec_shiloach_vishkin(UndirectedCSRGraph &_graph, int *_components);
    #endif

    #ifdef __USE_NEC_SX_AURORA__
    void nec_bfs_based(UndirectedCSRGraph &_graph, int *_components);
    #endif

    #ifdef __USE_GPU__
    void gpu_shiloach_vishkin(UndirectedCSRGraph &_graph, int *_components);
    #endif

    void seq_bfs_based(UndirectedCSRGraph &_graph, int *_components);

    double get_performance() { return performance; };
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "cc.hpp"
#include "nec_shiloach_vishkin.hpp"
#include "nec_bfs_based.hpp"
#include "seq_bfs_based.hpp"
#include "gpu_shiloach_vishkin.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
