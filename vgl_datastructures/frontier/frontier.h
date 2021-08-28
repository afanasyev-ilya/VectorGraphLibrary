#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "containers/base_frontier.h"
#include "containers/vect_csr/frontier_vect_csr.h"
#include "containers/csr/frontier_csr.h"
#include "containers/csr_vg/frontier_csr_vg.h"
#include "containers/edges_list/frontier_edges_list.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class VGL_Frontier
{
protected:
    ObjectType object_type;

    BaseFrontier *container;
public:
    /* constructors and destructors */
    VGL_Frontier(VGL_Graph &_graph, TraversalDirection _direction = ORIGINAL);
    ~VGL_Frontier();

    // get API
    inline int size() { return container->get_size(); };
    inline int get_size() { return container->get_size(); };
    inline int *get_ids() { return container->get_ids(); };
    inline int *get_flags() { return container->get_flags(); };
    inline FrontierSparsityType get_sparsity_type() { return container->get_sparsity_type(); };
    inline FrontierClassType get_class_type() { return container->get_class_type(); };
    inline long long get_neighbours_count() { return container->get_neighbours_count(); };

    BaseFrontier *get_container_data() { return container; };

    // printing API
    inline void print_stats() { container->print_stats(); };
    inline void print() { container->print(); };

    // frontier modification API
    inline void add_vertex(int _src_id) { container->add_vertex(_src_id); };
    inline void add_group_of_vertices(int *_vertex_ids, int _number_of_vertices){ container->add_group_of_vertices(_vertex_ids, _number_of_vertices); };
    inline void clear() { container->clear(); };
    inline void set_all_active() { container->set_all_active(); };

    // frontier direction API
    inline TraversalDirection get_direction() { return container->get_direction(); };
    inline void set_direction(TraversalDirection _direction) { container->set_direction(_direction); }; // TODO REMOVE
    inline void reorder(TraversalDirection _direction) { container->reorder(_direction); };

    #ifdef __USE_GPU__
    void move_to_host() { container->move_to_host(); };
    void move_to_device() { container->move_to_device(); };
    #endif
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "frontier.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
