#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../../framework_types.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
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

    FrontierNEC(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph);

    FrontierNEC(VectCSRGraph<_TVertexValue, _TEdgeWeight> &_graph);

    FrontierNEC(int _vertices_count);

    ~FrontierNEC();

    int size() {return current_size;};
    FrontierType get_type() {return type;};

    void set_all_active();

    void change_size(int _size) {max_size = _size;};

    void print_frontier_info(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph);

    inline void add_vertex(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int src_id);

    inline void add_vertices(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int *_vertex_ids, int _number_of_vertices);

    inline void clear() { current_size = 0; };

    friend class GraphPrimitivesNEC;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "frontier_nec.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
