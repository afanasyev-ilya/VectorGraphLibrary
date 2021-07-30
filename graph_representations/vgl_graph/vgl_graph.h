#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class VGL_Graph: public BaseGraph
{
private:
    UndirectedGraph *outgoing_data;
    UndirectedGraph *incoming_data;
public:
    VGL_Graph(GraphType _container_type);
    void import(EdgesContainer &_edges_container);

    /* get/set API */
    GraphType get_container_type() { return outgoing_data->get_type(); };
    UndirectedGraph *get_direction_data(TraversalDirection _direction);
    UndirectedGraph *get_outgoing_data() { return outgoing_data; };
    UndirectedGraph *get_incoming_data() { return incoming_data; };
    bool outgoing_is_stored() { return true; }; // TODO FIX

    /* print API */
    void print();
    void print_size();
    size_t get_size() { return outgoing_data->get_size() + incoming_data->get_size(); };

    /* file load/store API */
    bool save_to_binary_file(string _file_name) { return false; };
    bool load_from_binary_file(string _file_name)  { return false; };

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
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "vgl_graph.hpp"
#include "reorder.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
