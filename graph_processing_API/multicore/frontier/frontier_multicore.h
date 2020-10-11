#pragma once

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

    FrontierMulticore(UndirectedCSRGraph &_graph);
    FrontierMulticore(int _vertices_count);
    ~FrontierMulticore();

    int size() {return current_size;};
    FrontierType get_type() {return type;};

    void set_all_active();

    void change_size(int _size) {max_size = _size;};


    void print_stats(UndirectedCSRGraph &_graph);


    inline void add_vertex(UndirectedCSRGraph &_graph, int src_id);


    inline void add_group_of_vertices(UndirectedCSRGraph &_graph, int *_vertex_ids, int _number_of_vertices);

    inline void clear() { current_size = 0; };

    friend class GraphPrimitivesMulticore;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "frontier_multicore.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
