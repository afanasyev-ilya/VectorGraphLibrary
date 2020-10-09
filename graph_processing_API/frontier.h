#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Frontier
{
private:
    int *flags;
    int *ids;

    FrontierType type;

    int current_size;
    int max_size;
public:
    int size() {return current_size;};
    FrontierType get_type() {return type;};

    void set_all_active() {};

    void change_size(int _size) {max_size = _size;};


    void print_frontier_info(UndirectedGraph &_graph) {};


    inline void add_vertex(UndirectedGraph &_graph, int src_id) {};


    inline void add_vertices(UndirectedGraph &_graph,
                                     int *_vertex_ids,
                                     int _number_of_vertices) {};

    inline void clear() { current_size = 0; };
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

string get_frontier_status_string(FrontierType _type)
{
    string status;
    if(_type == ALL_ACTIVE_FRONTIER)
        status = "all active";
    if(_type == SPARSE_FRONTIER)
        status = "sparse";
    if(_type == DENSE_FRONTIER)
        status = "dense";

    return status;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "nec/frontier/frontier_nec.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
