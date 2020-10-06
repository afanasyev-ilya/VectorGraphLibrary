#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <vector>
#include <algorithm>
#include <string>
#include <fstream>
#include <stdio.h>

#include "../../common/cmd_parser/parser_options.h"
#include "../common/tmp_edge_data.h"
#include "vector_extension/vector_extension.h"
#include "../../common/memory_API/memory_API.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define VECTOR_EXTENSION_SIZE 7

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
class ExtendedCSRGraph : public BaseGraph<_TVertexValue, _TEdgeWeight>
{
private:
    long long     *vertex_pointers;
    int           *adjacent_ids;
    _TEdgeWeight  *adjacent_weights;

    int *forward_conversion;
    int *backward_conversion;

    VectorExtension<_TVertexValue, _TEdgeWeight> last_vertices_ve;
    
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

    void extract_connection_count(EdgesListGraph<_TVertexValue, _TEdgeWeight> &_el_graph,
                                  int *_work_buffer, int *_connections_count);

    void sort_vertices_by_degree(int *_connections_array, asl_int_t *_asl_indexes,
                                 int _el_vertices_count, int *_forward_conversion,
                                 int *_backward_conversion);

    void construct_CSR(EdgesListGraph<_TVertexValue, _TEdgeWeight> &_el_graph);
public:
    ExtendedCSRGraph(int _vertices_count = 1, long long _edges_count = 1);
    ~ExtendedCSRGraph();
    
    void resize(int _vertices_count, long long _edges_count);
    
    void print();
    //void print_with_weights(EdgesArrayNec<_TVertexValue, _TEdgeWeight, _TEdgeWeight> &_weights, TraversalDirection _direction);
    void print_stats() {};
    
    void save_to_graphviz_file(string _file_name, VisualisationMode _visualisation_mode = VISUALISE_AS_DIRECTED);
    bool save_to_binary_file(string file_name);
    bool load_from_binary_file(string file_name);
    
    void import_graph(EdgesListGraph<_TVertexValue, _TEdgeWeight> &_copy_graph,
                      VerticesState _vertices_state = VERTICES_SORTED,
                      EdgesState _edges_state = EDGES_SORTED,
                      int _supported_vector_length = 1,
                      AlgorithmTraversalType _traversal_type = PULL_TRAVERSAL,
                      MultipleArcsState _multiple_arcs_state = MULTIPLE_ARCS_PRESENT);

    void import_and_preprocess(EdgesListGraph<_TVertexValue, _TEdgeWeight> &_old_graph);
    
    inline int           *get_reordered_vertex_ids() {return reordered_vertex_ids;};
    inline long long     *get_vertex_pointers()        {return vertex_pointers;};
    inline int           *get_adjacent_ids()         {return adjacent_ids;};
    inline _TEdgeWeight  *get_adjacent_weights()     {return adjacent_weights;};

    inline long long get_edges_count_in_ve() {return last_vertices_ve.get_edges_count_in_ve();}

    inline VectorExtension<_TVertexValue, _TEdgeWeight>* get_ve_ptr() {return &last_vertices_ve;};

    inline long long get_csr_edge_id(int _src_id, int _dst_id);
    inline long long get_ve_edge_id(int _src_id, int _dst_id) { return last_vertices_ve.get_ve_edge_id(_src_id, _dst_id); };

    // renumber API
    int renumber_vertex_id(int _id); // TODO rename/rework
    void renumber_vertex_array(float *_input_array, float *_output_array); // TODO rename/rework
    
    #ifdef __USE_GPU__
    void move_to_device();
    void move_to_host();
    #endif
    
    #ifdef __USE_GPU__
    inline int get_gpu_grid_threshold_vertex(){return gpu_grid_threshold_vertex;};
    inline int get_gpu_block_threshold_vertex(){return gpu_block_threshold_vertex;};
    inline int get_gpu_warp_threshold_vertex(){return gpu_warp_threshold_vertex;};
    #endif

    #ifdef __USE_NEC_SX_AURORA__
    inline int get_vector_engine_threshold_vertex(){return vector_engine_threshold_vertex;};
    inline int get_vector_core_threshold_vertex(){return vector_core_threshold_vertex;};
    #endif

    template <typename _T>
    _T& get_edge_data(_T *_data_array, int _src_id, int _dst_id); // TODO remove

    void set_vertex_data_from_array(_TVertexValue *_values_array);

    size_t get_graph_size_in_bytes();
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define LOAD_EXTENDED_CSR_GRAPH_DATA(input_graph)                        \
int vertices_count                   = input_graph.get_vertices_count(); \
long long int edges_count            = input_graph.get_edges_count   (); \
\
long long    *vertex_pointers           = input_graph.get_vertex_pointers   ();\
int          *adjacent_ids            = input_graph.get_adjacent_ids    ();\
_TEdgeWeight *adjacent_weights        = input_graph.get_adjacent_weights();\
\
int ve_vertices_count = (input_graph.get_ve_ptr())->get_vertices_count();\
int ve_starting_vertex = (input_graph.get_ve_ptr())->get_starting_vertex();\
int ve_vector_segments_count = (input_graph.get_ve_ptr())->get_vector_segments_count();\
\
long long *ve_vector_group_ptrs = (input_graph.get_ve_ptr())->get_vector_group_ptrs();\
int *ve_vector_group_sizes = (input_graph.get_ve_ptr())->get_vector_group_sizes();\
int *ve_adjacent_ids = (input_graph.get_ve_ptr())->get_adjacent_ids();\
_TEdgeWeight *ve_adjacent_weights = (input_graph.get_ve_ptr())->get_adjacent_weights();\

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "extended_CSR_graph.hpp"
#include "preprocess.hpp"
#include "gpu_api.hpp"
#include "nec_api.hpp"
#include "renumber.hpp"
#include "print.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

