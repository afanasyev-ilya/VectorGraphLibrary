#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class BaseFrontier
{
protected:
    FrontierClassType class_type;

    int size;
    VGL_Graph *graph_ptr;
    TraversalDirection direction;

    int neighbours_count;

    // all frontiers have flags and ids - ?
    int *flags;
    int *ids;
    FrontierSparsityType sparsity_type;
public:
    BaseFrontier(VGL_Graph &_graph, TraversalDirection _direction)
    {
        graph_ptr = &_graph; direction = _direction; class_type = BASE_FRONTIER;
    };

    // get API
    inline int get_size() { return size; };
    inline int *get_ids() { return ids; };
    inline int *get_flags() { return flags; };
    inline FrontierSparsityType get_sparsity_type() { return sparsity_type; };
    inline FrontierClassType get_class_type() { return class_type; };
    inline long long get_neighbours_count() { return neighbours_count; };

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
    void reorder(TraversalDirection _direction) { set_direction(_direction); };
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
