#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <vector>
#include <algorithm>
#include <string>
#include <fstream>
#include <stdio.h>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "common/cmd_parser/parser_options.h"
#include "common/memory_API/memory_API.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class VectCSRGraph : public BaseGraph
{
private:
    UndirectedCSRGraph *outgoing_graph;
    UndirectedCSRGraph *incoming_graph;

    long long *vertices_reorder_buffer;

    void init(int _vertices_count, long long _edges_count);
    void free();
    void resize(int _vertices_count, long long _edges_count);

    // vertex reorder API (used in GraphAbstractions and VerticesArray)
    template <typename _T>
    bool vertices_buffer_can_be_used(VerticesArray<_T> &_data);
    template <typename _T>
    void reorder_to_original(VerticesArray<_T> &_data);
    template <typename _T>
    void reorder_to_scatter(VerticesArray<_T> &_data);
    template <typename _T>
    void reorder_to_gather(VerticesArray<_T> &_data);
    // allows to reorder verticesArray in arbitrary direction
    template <typename _T>
    void reorder(VerticesArray<_T> &_data, TraversalDirection _output_dir);

    // edges reorder API (used in Graph import and EdgesArray)
    // allows to reorder edges array to primary direction (scatter, from original)
    template <typename _T>
    void reorder_edges_original_to_scatter(_T *_original_data, _T *_outgoing_data);
    // allows to reorder edges array to secondary direction (gather, from original)
    template <typename _T>
    void reorder_edges_scatter_to_gather(_T *_gather_data, _T *_scatter_data);
public:
    VectCSRGraph(SupportedDirection _supported_direction = USE_BOTH,
                 int _vertices_count = 1, long long _edges_count = 1);
    ~VectCSRGraph();

    /* get/set API */
    long long get_direction_shift() {return (this->edges_count + this->get_edges_count_in_outgoing_ve());};

    bool outgoing_is_stored() { return can_use_scatter(); };
    bool incoming_is_stored() { return can_use_gather(); };

    inline int get_edge_dst(int _src_id, int _local_edge_pos, TraversalDirection _direction);
    inline int get_incoming_edge_dst(int _src_id, int _local_edge_pos);
    inline int get_outgoing_edge_dst(int _src_id, int _local_edge_pos);

    inline int get_connections_count(int _src_id, TraversalDirection _direction);
    inline int get_incoming_connections_count(int _src_id);
    inline int get_outgoing_connections_count(int _src_id);

    #ifdef __USE_MPI__
    std::pair<int, int> get_mpi_thresholds(int _mpi_rank, TraversalDirection _direction);
    #endif

    /* print API */
    void print();
    void print_size();
    void print_stats();
    size_t get_size();
    template <typename _T>
    void print_with_weights(EdgesArray<_T> &_weights);
    void print_vertex_information(TraversalDirection _direction, int _src_id, int _num_edges);

    /* file load/store API */
    bool save_to_binary_file(string file_name);
    bool load_from_binary_file(string file_name);

    /* GPU specific (copy) API */
    #ifdef __USE_GPU__
    void move_to_device();
    void move_to_host();
    #endif

    /* Further - VectCSRGraph specific API : reorder, working with double-directions, etc.*/

    // get pointers to the specific undirected part of graph (incoming or outgoing ids)
    UndirectedCSRGraph *get_outgoing_graph_ptr();
    UndirectedCSRGraph *get_incoming_graph_ptr();
    UndirectedCSRGraph *get_direction_graph_ptr(TraversalDirection _direction);

    // allows to get vector engine size
    inline long long get_edges_count_in_outgoing_ve();
    inline long long get_edges_count_in_incoming_ve();
    inline long long get_edges_count_in_outgoing_csr();
    inline long long get_edges_count_in_incoming_csr();

    /* reorder API */
    // allows to reorder a single vertex ID in arbitrary direction
    int reorder(int _vertex_id, TraversalDirection _input_dir, TraversalDirection _output_dir);

    // selects random vertex with non-zero outgoing and incoming degree
    int select_random_vertex(TraversalDirection _direction = ORIGINAL);

    // performs simple graph visualization using GraphViz API
    template <typename _TVertexValue>
    void save_to_graphviz_file(string _file_name, VerticesArray<_TVertexValue> &_vertex_data);

    /* import and preprocess API */
    // creates VectCSRGraph format from EdgesListGraph
    void import(EdgesListGraph &_copy_graph);

    template<class _T>
    friend class VerticesArray;
    template<class _T>
    friend class EdgesArray;
    template<class _T>
    friend class EdgesArray_EL;
    template<class _T>
    friend class EdgesArray_Vect;
    template<class _T>
    friend class EdgesArray_Sharded;
    friend class GraphAbstractions;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "vect_csr_graph.hpp"
#include "reorder.hpp"
#include "print.hpp"
#include "import.hpp"
#include "get_api.hpp"
#include "gpu_api.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

