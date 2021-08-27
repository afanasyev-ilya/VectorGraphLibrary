#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define BOTH_DIRECTIONS 2

class VGL_Graph: public BaseGraph
{
private:
    UndirectedGraph *outgoing_data;
    UndirectedGraph *incoming_data;

    void create_containers(GraphStorageFormat _container_type, GraphStorageOptimizations _optimizations);

    long long *vertices_reorder_buffer;

    // vertex reorder API (used in GraphAbstractions and VerticesArray)
    template <typename _T>
    bool vertices_buffer_can_be_used(VerticesArray<_T> &_data);

    template <typename _T>
    void reorder_to_original(VerticesArray<_T> &_data);
    template <typename _T>
    void reorder_to_scatter(VerticesArray<_T> &_data);
    template <typename _T>
    void reorder_to_gather(VerticesArray<_T> &_data);
public:
    VGL_Graph(GraphStorageFormat _container_type, GraphStorageOptimizations _optimizations = OPT_NONE);
    ~VGL_Graph();
    void import(EdgesContainer &_edges_container);

    /* get/set API */
    GraphStorageFormat get_container_type() { return outgoing_data->get_type(); };
    UndirectedGraph *get_direction_data(TraversalDirection _direction);
    UndirectedGraph *get_outgoing_data() { return outgoing_data; };
    UndirectedGraph *get_incoming_data() { return incoming_data; };
    bool outgoing_is_stored() { return true; }; // TODO FIX

    inline int get_edge_dst(int _src_id, int _local_edge_pos, TraversalDirection _direction);
    inline int get_incoming_edge_dst(int _src_id, int _local_edge_pos);
    inline int get_outgoing_edge_dst(int _src_id, int _local_edge_pos);

    inline int get_connections_count(int _src_id, TraversalDirection _direction);
    inline int get_incoming_connections_count(int _src_id);
    inline int get_outgoing_connections_count(int _src_id);

    inline size_t get_outgoing_edges_array_index(int _v, int _edge_pos);
    inline size_t get_incoming_edges_array_index(int _v, int _edge_pos);

    inline int get_number_of_directions() {return BOTH_DIRECTIONS;};
    inline size_t get_edges_array_direction_shift_size() {return outgoing_data->get_edges_array_direction_shift_size();};

    /* print API */
    void print();
    void print_size();
    size_t get_size() { return outgoing_data->get_size() + incoming_data->get_size(); };

    /* file load/store API */
    bool save_to_binary_file(string _file_name);
    bool load_from_binary_file(string _file_name);

    /* reorder API */
    template <typename _T>
    void reorder(VerticesArray<_T> &_data, TraversalDirection _output_dir);
    int reorder(int _vertex_id, TraversalDirection _input_dir, TraversalDirection _output_dir);
    int select_random_vertex(TraversalDirection _direction = ORIGINAL);
    int select_random_nz_vertex(TraversalDirection _direction = ORIGINAL);

    template <typename _T>
    void copy_outgoing_to_incoming_edges(_T *_outgoing_edges, _T *_incoming_edges);

    template <typename _T>
    void copy_incoming_to_outgoing_edges(_T *_outgoing_edges, _T *_incoming_edges);

    /* GPU specific (copy) API */
    #ifdef __USE_GPU__
    void move_to_device() final;
    void move_to_host() final;
    #endif
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "vgl_graph.hpp"
#include "reorder.hpp"
#include "get_api.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
