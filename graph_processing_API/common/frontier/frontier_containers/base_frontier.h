#pragma once

class BaseFrontier
{
protected:
    int size;
    VGL_Graph *graph_ptr;
    TraversalDirection direction;

    // all frontiers have flags and ids - ?
    int *flags;
    int *ids;
    FrontierType type;
public:
    BaseFrontier(VGL_Graph &_graph, TraversalDirection _direction)
    {
        graph_ptr = &_graph; direction = _direction;
    };

    // get API
    inline int get_size() { return size; };
    inline int *get_ids() { return ids; };
    inline int *get_flags() { return flags; };
    inline FrontierType get_type() { return type; };

    // printing API
    virtual void print_stats() = 0;
    virtual void print() = 0;

    // frontier modification API
    virtual void add_vertex(int _src_id) = 0;
    virtual void add_group_of_vertices(int *_vertex_ids, int _number_of_vertices) = 0;
    virtual void clear() = 0;

    // modification
    virtual void set_all_active() = 0;

    // frontier direction API
    TraversalDirection get_direction() { return direction; };
    void set_direction(TraversalDirection _direction) { direction = _direction; };
    void reorder(TraversalDirection _direction) { throw " FRONTIER reorder not implemented yet"; };
};