#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define FRONTIER_TYPE_CHANGE_THRESHOLD 0.1
#define VE_FRONTIER_TYPE_CHANGE_THRESHOLD 0.01
#define VC_FRONTIER_TYPE_CHANGE_THRESHOLD 0.01
#define COLLECTIVE_FRONTIER_TYPE_CHANGE_THRESHOLD 0.05

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../../framework_types.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class FrontierNEC
{
private:
    int *flags;
    int *ids;
    int *work_buffer;

    FrontierType type;

    int vector_engine_part_size;
    int vector_core_part_size;
    int collective_part_size;

    FrontierType vector_engine_part_type;
    FrontierType vector_core_part_type;
    FrontierType collective_part_type;

    int current_size;
    int max_size;
public:
    template <typename _TVertexValue, typename _TEdgeWeight>
    FrontierNEC(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph);
    FrontierNEC(int _vertices_count);
    ~FrontierNEC();

    int size() {return current_size;};
    FrontierType get_type() {return type;};

    void set_all_active();

    void change_size(int _size) {max_size = _size;};

    template <typename _TVertexValue, typename _TEdgeWeight>
    void print_frontier_info(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph);

    template <typename _TVertexValue, typename _TEdgeWeight>
    void add_vertex(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int src_id);

    friend class GraphPrimitivesNEC;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "frontier_nec.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
