#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../../framework_types.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class FrontierMulticore
{
private:
    int *flags;
    int *ids;

    FrontierType type;

    int current_size;
    int max_size;
public:
    template <typename _TVertexValue, typename _TEdgeWeight>
    FrontierMulticore(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph);
    FrontierMulticore(int _vertices_count);
    ~FrontierMulticore();

    int size() {return current_size;};
    FrontierType get_type() {return type;};

    void set_all_active();

    void change_size(int _size) {max_size = _size;};

    template <typename _TVertexValue, typename _TEdgeWeight>
    void print_frontier_info(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph);

    template <typename _TVertexValue, typename _TEdgeWeight>
    inline void add_vertex(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int src_id);

    template <typename _TVertexValue, typename _TEdgeWeight>
    inline void add_vertices(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int *_vertex_ids, int _number_of_vertices);

    inline void clear() { current_size = 0; };

    friend class GraphPrimitivesMulticore;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "frontier_multicore.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
