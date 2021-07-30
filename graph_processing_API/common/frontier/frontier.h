#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "frontier_containers/base_frontier.h"
#include "frontier_containers/vect_csr/frontier_vect_csr.h"

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

    // printing API
    inline void print_stats() { frontier_representation->print_stats(); };
    inline void print() { frontier_representation->print(); };

    // frontier modification API
    inline void add_vertex(int _src_id) { frontier_representation->add_vertex(_src_id); };
    inline void clear() { frontier_representation->clear(); };

    // frontier direction API
    inline TraversalDirection get_direction() { return frontier_representation->get_direction(); };
    inline void set_direction(TraversalDirection _direction) { frontier_representation->set_direction(_direction); }; // TODO REMOVE
    inline void reorder(TraversalDirection _direction) { frontier_representation->reorder(_direction); };
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "frontier.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
#ifdef __USE_NEC_SX_AURORA__
#include "graph_processing_API/nec/frontier/frontier_nec.h"
#endif

#if defined(__USE_GPU__)
#include "graph_processing_API/gpu/frontier/frontier_gpu.cuh"
#endif

#if defined(__USE_MULTICORE__) || defined(__USE_MULTICORE__)
#include "graph_processing_API/multicore/frontier/frontier_multicore.h"
#endif*/

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
