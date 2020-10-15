#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <map>
#include <iterator>
#include <vector>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class ShardedCSRGraph : public BaseGraph
{
private:
    int shards_number;
    int max_cached_vertices;

    UndirectedCSRGraph *outgoing_shards;
    UndirectedCSRGraph *incoming_shards;

    long long *vertices_reorder_buffer;

    void import_direction(EdgesListGraph &_el_graph, UndirectedCSRGraph **_shards_ptr);

    int get_shard_id(int _dst_id) { return _dst_id / max_cached_vertices; };
    void resize_helper_arrays();
public:
    ShardedCSRGraph();
    ~ShardedCSRGraph();

    /* get API */
    inline int get_shards_number() {return shards_number;};
    inline UndirectedCSRGraph *get_outgoing_shard_ptr(int _shard_id) {return &(outgoing_shards[_shard_id]);};
    inline UndirectedCSRGraph *get_incoming_shard_ptr(int _shard_id) {return &(incoming_shards[_shard_id]);};

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

    /* reorder API */
    template <typename _T>
    void reorder_to_sorted_for_shard(VerticesArrayNEC<_T> _data, int _shard_id);
    template <typename _T>
    void reorder_to_original_for_shard(VerticesArrayNEC<_T> _data, int _shard_id);

    // creates ShardedCSRGraph format from EdgesListGraph
    void import(EdgesListGraph &_el_graph);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "sharded_csr_graph.hpp"
#include "import.hpp"
#include "reorder.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
