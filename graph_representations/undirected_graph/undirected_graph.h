#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <vector>
#include <algorithm>
#include <string>
#include <fstream>
#include <stdio.h>

#include "../../common/cmd_parser/parser_options.h"
#include "vector_extension/vector_extension.h"
#include "../../common/memory_API/memory_API.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define VECTOR_EXTENSION_SIZE 7
#define ATTEMPTS_THRESHOLD 100

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
class VerticesArrayNec;

template <typename _T>
class EdgesArrayNec;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class UndirectedGraph: public BaseGraph
{
private:
    long long     *vertex_pointers;
    int           *adjacent_ids;

    int *forward_conversion; // forward = to sorted, forward(i) = sorted
    int *backward_conversion; // backward = to original, backward(i) = original

    VectorExtension last_vertices_ve;
    
    #ifdef __USE_GPU__
    int gpu_grid_threshold_vertex;
    int gpu_block_threshold_vertex;
    int gpu_warp_threshold_vertex;
    #endif

    #ifdef __USE_NEC_SX_AURORA__
    int vector_engine_threshold_vertex;
    int vector_core_threshold_vertex;
    #endif

    void alloc(int _vertices_count, long long _edges_count);
    void free();

    #ifdef __USE_GPU__
    void estimate_gpu_thresholds();
    #endif

    #ifdef __USE_NEC_SX_AURORA__
    void estimate_nec_thresholds();
    #endif

    void extract_connection_count(EdgesListGraph &_el_graph,
                                  int *_work_buffer, int *_connections_count);

    void sort_vertices_by_degree(int *_connections_array, asl_int_t *_asl_indexes,
                                 int _el_vertices_count, int *_forward_conversion,
                                 int *_backward_conversion);

    void construct_CSR(EdgesListGraph &_el_graph);

    void copy_edges_indexes(long long *_edges_reorder_indexes, asl_int_t *_asl_indexes, long long _edges_count);
public:
    UndirectedGraph(int _vertices_count = 1, long long _edges_count = 1);
    ~UndirectedGraph();

    /* get API */
    inline long long       *get_vertex_pointers()        {return vertex_pointers;};
    inline int             *get_adjacent_ids()         {return adjacent_ids;};
    inline long long        get_edges_count_in_ve() {return last_vertices_ve.get_edges_count_in_ve();}
    inline VectorExtension* get_ve_ptr() {return &last_vertices_ve;};
    inline int get_connections_count(int _vertex_id) {return (vertex_pointers[_vertex_id+1] - vertex_pointers[_vertex_id]);};

    /* print API */
    void print();
    void print_size();
    size_t get_size();
    template <typename _T>
    void print_with_weights(EdgesArrayNec<_T> &_weights, TraversalDirection _direction);

    /* file load/store API */
    void save_to_graphviz_file(string _file_name, VisualisationMode _visualisation_mode = VISUALISE_AS_DIRECTED);
    bool save_to_binary_file(string file_name);
    bool load_from_binary_file(string file_name);

    /* GPU specific (copy) API */
    #ifdef __USE_GPU__
    void move_to_device();
    void move_to_host();
    #endif

    /* Further - undirectedGraph specific API : reorder, import, working with VEs etc.*/
    // resize graph
    void resize(int _vertices_count, long long _edges_count);

    // allows to get position of specified edge in CSR representation
    inline long long get_csr_edge_id(int _src_id, int _dst_id);

    // allows to get position of specified edge in VE representation
    inline long long get_ve_edge_id (int _src_id, int _dst_id) { return last_vertices_ve.get_ve_edge_id(_src_id, _dst_id); };

    // main function to create vector CSR format
    void import_and_preprocess(EdgesListGraph &_old_graph, long long *_edges_reorder_indexes);

    /* reorder API */
    // reorders a single vertex from original (edges list) to sorted (undirectedCSR)
    int reorder_to_sorted(int _vertex_id);

    // reorders a single vertex from sorted (undirectedCSR) to original (edges list)
    int reorder_to_original(int _vertex_id);

    // reorders a vertexArray(pointer) from original (edges list) to sorted (undirectedCSR)
    template <typename _T>
    void reorder_to_sorted(_T *_data, _T *_buffer);

    // reorders a vertexArray(pointer)  from sorted (undirectedCSR) to original (edges list)
    template <typename _T>
    void reorder_to_original(_T *_data, _T *_buffer);

    // API to calculate GPU thresholds // TODO remove
    #ifdef __USE_GPU__
    inline int get_gpu_grid_threshold_vertex(){return gpu_grid_threshold_vertex;};
    inline int get_gpu_block_threshold_vertex(){return gpu_block_threshold_vertex;};
    inline int get_gpu_warp_threshold_vertex(){return gpu_warp_threshold_vertex;};
    #endif

    // API to calculate NEC and multicore thresholds
    #ifdef __USE_NEC_SX_AURORA__
    inline int get_vector_engine_threshold_vertex(){return vector_engine_threshold_vertex;};
    inline int get_vector_core_threshold_vertex(){return vector_core_threshold_vertex;};
    #endif

    template <typename _T>
    _T& get_edge_data(_T *_data_array, int _src_id, int _dst_id); // TODO remove, needed in max Flow?

    // selects random vertex with non-zero degree
    int select_random_vertex();
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define LOAD_UNDIRECTED_CSR_GRAPH_DATA(input_graph)                        \
int vertices_count                   = input_graph.get_vertices_count(); \
long long int edges_count            = input_graph.get_edges_count   (); \
\
long long    *vertex_pointers           = input_graph.get_vertex_pointers   ();\
int          *adjacent_ids            = input_graph.get_adjacent_ids    ();\
\
int ve_vertices_count = (input_graph.get_ve_ptr())->get_vertices_count();\
int ve_starting_vertex = (input_graph.get_ve_ptr())->get_starting_vertex();\
int ve_vector_segments_count = (input_graph.get_ve_ptr())->get_vector_segments_count();\
\
long long *ve_vector_group_ptrs = (input_graph.get_ve_ptr())->get_vector_group_ptrs();\
int *ve_vector_group_sizes = (input_graph.get_ve_ptr())->get_vector_group_sizes();\
int *ve_adjacent_ids = (input_graph.get_ve_ptr())->get_adjacent_ids();\

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "undirected_graph.hpp"
#include "preprocess.hpp"
#include "gpu_api.hpp"
#include "nec_api.hpp"
#include "reorder.hpp"
#include "print.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

