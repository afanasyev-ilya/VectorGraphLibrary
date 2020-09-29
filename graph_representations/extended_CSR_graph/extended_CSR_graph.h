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

#define VECTOR_EXTENSION_SIZE 7

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
class ExtendedCSRGraph : public BaseGraph<_TVertexValue, _TEdgeWeight>
{
private:
VerticesState vertices_state;
    EdgesState edges_state;
    int supported_vector_length;
    
    int           *reordered_vertex_ids;
    long long     *outgoing_ptrs;
    int           *outgoing_ids;
    _TEdgeWeight  *outgoing_weights;

    long long     *incoming_ptrs;
    int           *incoming_ids;
    _TEdgeWeight  *incoming_weights;

    VectorExtension<_TVertexValue, _TEdgeWeight> last_vertices_ve;
    
    #ifdef __USE_GPU__
    int gpu_grid_threshold_vertex;
    int gpu_block_threshold_vertex;
    int gpu_warp_threshold_vertex;
    #endif

    //#ifdef __USE_NEC_SX_AURORA__
    int vector_engine_threshold_vertex;
    int vector_core_threshold_vertex;
    //#endif
    
    int *incoming_degrees;
    
    void alloc(int _vertices_count, long long _edges_count);
    void free();
    
    void calculate_incoming_degrees();
    
    #ifdef __USE_GPU__
    void estimate_gpu_thresholds();
    #endif

    //#ifdef __USE_NEC_SX_AURORA__
    void estimate_nec_thresholds();
    //#endif
public:
    ExtendedCSRGraph(int _vertices_count = 1, long long _edges_count = 1);
    ~ExtendedCSRGraph();
    
    void resize(int _vertices_count, long long _edges_count);
    
    void print();
    void print_stats() {};
    
    void save_to_graphviz_file(string _file_name, VisualisationMode _visualisation_mode = VISUALISE_AS_DIRECTED);
    bool save_to_binary_file(string file_name);
    bool load_from_binary_file(string file_name);
    
    void import_graph(EdgesListGraph<_TVertexValue, _TEdgeWeight> &_copy_graph,
                      VerticesState _vertices_state = VERTICES_SORTED,
                      EdgesState _edges_state = EDGES_SORTED,
                      int _supported_vector_length = 1,
                      TraversalDirection _traversal_type = PULL_TRAVERSAL,
                      MultipleArcsState _multiple_arcs_state = MULTIPLE_ARCS_PRESENT);
    
    inline int           *get_reordered_vertex_ids() {return reordered_vertex_ids;};
    inline long long     *get_outgoing_ptrs()        {return outgoing_ptrs;};
    inline int           *get_outgoing_ids()         {return outgoing_ids;};
    inline _TEdgeWeight  *get_outgoing_weights()     {return outgoing_weights;};
    inline int           *get_incoming_degrees()     {return incoming_degrees;};
    
    #ifdef __USE_GPU__
    void move_to_device();
    void move_to_host();
    #endif
    
    #ifdef __USE_GPU__
    inline int get_gpu_grid_threshold_vertex(){return gpu_grid_threshold_vertex;};
    inline int get_gpu_block_threshold_vertex(){return gpu_block_threshold_vertex;};
    inline int get_gpu_warp_threshold_vertex(){return gpu_warp_threshold_vertex;};
    #endif

    //#ifdef __USE_NEC_SX_AURORA__
    inline int get_vector_engine_threshold_vertex(){return vector_engine_threshold_vertex;};
    inline int get_vector_core_threshold_vertex(){return vector_core_threshold_vertex;};
    //#endif

    template <typename _T>
    _T& get_edge_data(_T *_data_array, int _src_id, int _dst_id);

    void set_vertex_data_from_array(_TVertexValue *_values_array);

    VectorExtension<_TVertexValue, _TEdgeWeight> *get_last_vertices_ve_ptr(){return &last_vertices_ve;}

    size_t get_graph_size_in_bytes();
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define LOAD_EXTENDED_CSR_GRAPH_DATA(input_graph)                        \
int vertices_count                   = input_graph.get_vertices_count(); \
long long int edges_count            = input_graph.get_edges_count   (); \
\
long long    *outgoing_ptrs           = _graph.get_outgoing_ptrs   ();\
int          *outgoing_ids            = _graph.get_outgoing_ids    ();\
_TEdgeWeight *outgoing_weights        = _graph.get_outgoing_weights();\
\
int ve_vertices_count = (_graph.get_last_vertices_ve_ptr())->get_vertices_count();\
int ve_starting_vertex = (_graph.get_last_vertices_ve_ptr())->get_starting_vertex();\
int ve_vector_segments_count = (_graph.get_last_vertices_ve_ptr())->get_vector_segments_count();\
\
long long *ve_vector_group_ptrs = (_graph.get_last_vertices_ve_ptr())->get_vector_group_ptrs();\
int *ve_vector_group_sizes = (_graph.get_last_vertices_ve_ptr())->get_vector_group_sizes();\
int *ve_outgoing_ids = (_graph.get_last_vertices_ve_ptr())->get_adjacent_ids();\
_TEdgeWeight *ve_outgoing_weights = (_graph.get_last_vertices_ve_ptr())->get_adjacent_weights();\
\
int *incoming_degrees = input_graph.get_incoming_degrees();\

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "extended_CSR_graph.hpp"
#include "init_graph.hpp"
#include "gpu_api.hpp"
#include "nec_api.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

