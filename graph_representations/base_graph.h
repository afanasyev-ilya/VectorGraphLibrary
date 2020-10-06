//
//  base_graph.h
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 14/04/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef base_graph_h
#define base_graph_h

#include "common/graph_types.h"
#include <string>

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum VisualisationMode
{
    VISUALISE_AS_DIRECTED = 0,
    VISUALISE_AS_UNDIRECTED = 1
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class BaseGraph
{
protected:
    GraphType graph_type;
    
    int vertices_count;
    long long edges_count;
    
    bool graph_on_device;
public:
    BaseGraph() {this->graph_on_device = false;};
    ~BaseGraph() {};
    
    inline int get_vertices_count() {return vertices_count;};
    inline long long get_edges_count() {return edges_count;};
    
    virtual void print() = 0;
    virtual void print_stats() = 0;
    virtual void save_to_graphviz_file(string file_name, VisualisationMode _visualisation_mode = VISUALISE_AS_DIRECTED) = 0;
    virtual bool save_to_binary_file(string file_name) = 0;
    virtual bool load_from_binary_file(string file_name) = 0;
    
    #ifdef __USE_GPU__
    virtual void move_to_device() = 0;
    virtual void move_to_host() = 0;
    #endif
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* base_graph_h */
