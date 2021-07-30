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

    /* print API */
    void print();
    void print_size() {};
    size_t get_size() { return 0; };

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
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "vgl_graph.hpp"
#include "reorder.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
