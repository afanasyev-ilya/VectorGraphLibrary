#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <map>
#include <iterator>
#include <vector>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class ShardedGraph : public BaseGraph
{
private:
    int shards_number;
    int max_cached_vertices;

    UndirectedCSRGraph *outgoing_shards;

    int get_shard_id(int _dst_id) { return _dst_id / max_cached_vertices; };
public:
    ShardedGraph();
    ~ShardedGraph();

    /* get API */
    inline int get_vertices_count() {return vertices_count;};
    inline long long get_edges_count() {return edges_count;};

    /* print API */
    void print() {};
    void print_size() {};
    size_t get_size() {return 0;};

    /* file load/store API */
    void save_to_graphviz_file(string file_name, VisualisationMode _visualisation_mode = VISUALISE_AS_DIRECTED) {};
    bool save_to_binary_file(string file_name) {return false;};
    bool load_from_binary_file(string file_name) {return false;};

    /* GPU specific (copy) API */
    #ifdef __USE_GPU__
    void move_to_device() {};
    void move_to_host() {};
    #endif

    /* Further - ShardedCSRGraph specific API : reorder, working with double-directions, etc.*/

    // creates ShardedCSRGraph format from EdgesListGraph
    void import(EdgesListGraph &_el_graph);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "sharded_csr_graph.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
