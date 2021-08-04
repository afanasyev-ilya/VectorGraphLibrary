#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class VGL_Graph: public BaseGraph
{
private:
    UndirectedGraph *outgoing_data;
    UndirectedGraph *incoming_data;

    void create_containers(GraphType _container_type);

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
    VGL_Graph(GraphType _container_type);
    ~VGL_Graph();
    void import(EdgesContainer &_edges_container);

    /* get/set API */
    GraphType get_container_type() { return outgoing_data->get_type(); };
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

    /* print API */
    void print();
    void print_size();
    size_t get_size() { return outgoing_data->get_size() + incoming_data->get_size(); };

    /* file load/store API */
    bool save_to_binary_file(string _file_name);
    bool load_from_binary_file(string _file_name);

    /* GPU specific (copy) API */
    #ifdef __USE_GPU__
    void move_to_device() {};
    void move_to_host() {};
    #endif

    /* reorder API */
    template <typename _T>
    void reorder(VerticesArray<_T> &_data, TraversalDirection _output_dir);
    int reorder(int _vertex_id, TraversalDirection _input_dir, TraversalDirection _output_dir);
    int select_random_vertex(TraversalDirection _direction = ORIGINAL);
    int select_random_nz_vertex(TraversalDirection _direction = ORIGINAL);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "vgl_graph.hpp"
#include "reorder.hpp"
#include "get_api.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
