#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "containers/base_frontier.h"
#include "containers/vect_csr/frontier_vect_csr.h"
#include "containers/general/frontier_general.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class VGL_Frontier
{
protected:
    ObjectType object_type;

    BaseFrontier *frontier_representation;
public:
    /* constructors and destructors */
    VGL_Frontier(VGL_Graph &_graph, TraversalDirection _direction = ORIGINAL);
    ~VGL_Frontier();

    // get API
    inline int size() { return frontier_representation->get_size(); };
    inline int get_size() { return frontier_representation->get_size(); };
    inline int *get_ids() { return frontier_representation->get_ids(); };
    inline int *get_flags() { return frontier_representation->get_flags(); };
    inline FrontierSparsityType get_sparsity_type() { return frontier_representation->get_sparsity_type(); };
    inline FrontierClassType get_class_type() { return frontier_representation->get_class_type(); };
    inline long long get_neighbours_count() { return frontier_representation->get_neighbours_count(); };

    BaseFrontier *get_container_data() { return frontier_representation; };

    // printing API
    inline void print_stats() { frontier_representation->print_stats(); };
    inline void print() { frontier_representation->print(); };

    // frontier modification API
    inline void add_vertex(int _src_id) { frontier_representation->add_vertex(_src_id); };
    inline void add_group_of_vertices(int *_vertex_ids, int _number_of_vertices){ frontier_representation->add_group_of_vertices(_vertex_ids, _number_of_vertices); };
    inline void clear() { frontier_representation->clear(); };
    inline void set_all_active() { frontier_representation->set_all_active(); };

    // frontier direction API
    inline TraversalDirection get_direction() { return frontier_representation->get_direction(); };
    inline void set_direction(TraversalDirection _direction) { frontier_representation->set_direction(_direction); }; // TODO REMOVE
    inline void reorder(TraversalDirection _direction) { frontier_representation->reorder(_direction); };
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "frontier.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
