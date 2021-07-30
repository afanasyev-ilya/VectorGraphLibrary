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

    VectorCSRGraph *outgoing_shards;
    VectorCSRGraph *incoming_shards;

    long long *vertices_reorder_buffer;

    void import_direction_2D_segmented(EdgesListGraph &_el_graph, TraversalDirection _import_direction);
    void import_direction_random_segmenting(EdgesListGraph &_el_graph, TraversalDirection _import_direction);

    int get_shard_id(int _dst_id) { return _dst_id / max_cached_vertices; };

    void resize(int _shards_count, int _vertices_count);
    void init(int _shards_count, int _vertices_count);
    void free();

    /* reorder API */
    template <typename _T>
    void reorder_from_original_to_shard(VerticesArray<_T> _data, TraversalDirection _direction, int _shard_id);
    template <typename _T>
    void reorder_from_shard_to_original(VerticesArray<_T> _data, TraversalDirection _direction, int _shard_id);
public:
    ShardedCSRGraph(SupportedDirection _supported_direction = USE_BOTH);
    ~ShardedCSRGraph();

    /* get API */
    inline int get_shards_number() {return shards_number;};
    inline VectorCSRGraph *get_outgoing_shard_ptr(int _shard_id) {return &(outgoing_shards[_shard_id]);};
    inline VectorCSRGraph *get_incoming_shard_ptr(int _shard_id) {return &(incoming_shards[_shard_id]);};
    inline VectorCSRGraph *get_shard_ptr(int _shard_id, TraversalDirection _direction);

    inline long long get_edges_count_outgoing_shard(int _shard_id) {return outgoing_shards[_shard_id].get_edges_count();};
    inline long long get_edges_count_incoming_shard(int _shard_id) {return incoming_shards[_shard_id].get_edges_count();};
    inline long long get_edges_count_in_ve_outgoing_shard(int _shard_id) {return outgoing_shards[_shard_id].get_edges_count_in_ve();};
    inline long long get_edges_count_in_ve_incoming_shard(int _shard_id) {return incoming_shards[_shard_id].get_edges_count_in_ve();};

    inline long long get_direction_shift();
    inline long long get_shard_shift(int _shard_id, TraversalDirection _direction);

    bool outgoing_is_stored() { return can_use_scatter(); };
    bool incoming_is_stored() { return can_use_gather(); };

    /* reorder API */
    // allows to reorder a single vertex ID in arbitrary direction
    int reorder(int _vertex_id, TraversalDirection _input_dir, TraversalDirection _output_dir);

    /* print API */
    void print();
    void print_size();
    template <typename _T>
    void print_in_csr_format(EdgesArray_Sharded<_T> &_weights);
    size_t get_size();

    /* file load/store API */
    void save_to_graphviz_file(string _file_name, VisualisationMode _visualisation_mode = VISUALISE_AS_DIRECTED) {};
    bool save_to_binary_file(string _file_name);
    bool load_from_binary_file(string _file_name);

    /* GPU specific (copy) API */
    #ifdef __USE_GPU__
    void move_to_device() {};
    void move_to_host() {};
    #endif

    /* Further - ShardedCSRGraph specific API : reorder, working with double-directions, etc.*/

    // creates ShardedCSRGraph format from EdgesListGraph
    void import(EdgesListGraph &_el_graph, int _force_shards = 0);

    // selects random vertex with non-zero outgoing and incoming degree
    int select_random_vertex(TraversalDirection _direction = ORIGINAL);

    template <typename _T> friend class VerticesArray;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "sharded_csr_graph.hpp"
#include "import.hpp"
#include "reorder.hpp"
#include "print.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
