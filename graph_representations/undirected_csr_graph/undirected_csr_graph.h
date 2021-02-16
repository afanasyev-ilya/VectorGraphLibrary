#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <vector>
#include <algorithm>
#include <string>
#include <fstream>
#include <stdio.h>
#include <string.h>

#include "common/cmd_parser/parser_options.h"
#include "vector_extension/vector_extension.h"
#include "common/memory_API/memory_API.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define VECTOR_EXTENSION_SIZE 7
#define ATTEMPTS_THRESHOLD 100

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
class VerticesArray;

template <typename _T>
class EdgesArray;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class UndirectedCSRGraph: public BaseGraph
{
private:
    long long     *vertex_pointers;
    int           *adjacent_ids;

    int *forward_conversion; // forward = to sorted, forward(i) = sorted
    int *backward_conversion; // backward = to original, backward(i) = original

    vgl_sort_indexes *edges_reorder_indexes; // allows to convert UndirectedCSRGraph edges (and weights) from sorted to original order

    VectorExtension last_vertices_ve; // store last vertices in the vector extension
    
    #ifdef __USE_GPU__
    int gpu_grid_threshold_vertex;
    int gpu_block_threshold_vertex;
    int gpu_warp_threshold_vertex;
    #endif

    #if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
    int vector_engine_threshold_vertex;
    int vector_core_threshold_vertex;
    #endif

    void alloc(int _vertices_count, long long _edges_count);
    void free();

    #if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
    void estimate_nec_thresholds();
    #endif

    // import functions
    void extract_connection_count(EdgesListGraph &_el_graph,
                                  int *_work_buffer, int *_connections_count);

    void sort_vertices_by_degree(int *_connections_array, vgl_sort_indexes *_sort_indexes,
                                 int _el_vertices_count, int *_forward_conversion,
                                 int *_backward_conversion);

    void construct_CSR(EdgesListGraph &_el_graph);

    void remove_loops_and_multiple_arcs();

    void copy_edges_indexes(vgl_sort_indexes *_sort_indexes);

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
    // allows to save edge reorder indexes
    void update_edge_reorder_indexes_using_superposition(vgl_sort_indexes *_edges_reorder_indexes);
    // in-place edges reorder from original to sorted, buffer can be provided for better speed
    template <typename _T>
    void reorder_edges_to_sorted(_T *_data, _T *_buffer = NULL);
    // in-place edges reorder from sorted to original, buffer can be provided for better speed
    template <typename _T>
    void reorder_edges_to_original(_T *_data, _T *_buffer = NULL);
    // allows to copy data from original (usually EdgesList weights) to sorted. Original array can be larger (used in sharded API).
    template <typename _T>
    void reorder_and_copy_edges_from_original_to_sorted(_T *_dst_sorted, _T *_src_original);

    // allows to get position of specified edge in CSR representation
    inline long long get_csr_edge_id(int _src_id, int _dst_id);

    // allows to get position of specified edge in VE representation
    inline long long get_ve_edge_id (int _src_id, int _dst_id) { return last_vertices_ve.get_ve_edge_id(_src_id, _dst_id); };

    void save_main_content_to_binary_file(FILE *_graph_file);
    void load_main_content_to_binary_file(FILE *_graph_file);
public:
    UndirectedCSRGraph(int _vertices_count = 1, long long _edges_count = 1);
    ~UndirectedCSRGraph();

    /* get API */
    inline long long       *get_vertex_pointers()        {return vertex_pointers;};
    inline int             *get_adjacent_ids()         {return adjacent_ids;};
    inline long long        get_edges_count_in_ve() {return last_vertices_ve.get_edges_count_in_ve();}
    inline VectorExtension* get_ve_ptr() {return &last_vertices_ve;};
    inline int get_connections_count(int _vertex_id) {return (vertex_pointers[_vertex_id+1] - vertex_pointers[_vertex_id]);};

    /* print API */
    void print();
    void print_size();
    void print_stats();
    size_t get_size();
    size_t get_csr_size();
    size_t get_ve_size();
    template <typename _T>
    void print_with_weights(EdgesArray<_T> &_weights, TraversalDirection _direction);
    void print_vertex_information(int _src_id, int _num_edges);

    /* file load/store API */
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

    // API to calculate NEC and multicore thresholds
    #if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
    inline int get_vector_engine_threshold_vertex(){return vector_engine_threshold_vertex;};
    inline int get_vector_core_threshold_vertex(){return vector_core_threshold_vertex;};
    #endif

    // selects random vertex with non-zero degree
    int select_random_vertex();

    // performs simple graph visualization using GraphViz API
    template <typename _TVertexValue>
    void save_to_graphviz_file(string _file_name, VerticesArray<_TVertexValue> &_vertex_data,
                               VisualisationMode _visualisation_mode = VISUALISE_AS_DIRECTED);

    /* import and preprocess API */
    // creates UndirectedCSRGraph format from EdgesListGraph
    void import(EdgesListGraph &_old_graph);

    void sort_adjacent_edges();

    friend class GraphAbstractions;
    friend class VectCSRGraph;
    friend class ShardedCSRGraph;
    friend class EdgesListGraph;
    template<class _T> friend class EdgesArray_Vect;
    template<class _T> friend class EdgesArray_Sharded;
    template<class _T> friend class EdgesArray_EL;
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

#include "undirected_csr_graph.hpp"
#include "import.hpp"
#include "gpu_api.hpp"
#include "nec_api.hpp"
#include "reorder.hpp"
#include "print.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

