//
//  vectorised_CSR_graph.h
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 14/04/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef extended_CSR_graph_h
#define extended_CSR_graph_h

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <vector>
#include <algorithm>
#include <string>
#include <fstream>
#include <stdio.h>

#include "../common/vectorise_CSR.h"

#define VECTOR_EXTENSION_SIZE 7

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
class ExtendedCSRGraph : public BaseGraph<_TVertexValue, _TEdgeWeight>
{
private:
    int threads_count;
    
    VerticesState vertices_state;
    EdgesState edges_state;
    int supported_vector_length;
    
    int           *reordered_vertex_ids;
    long long     *outgoing_ptrs;
    int           *outgoing_ids;
    _TEdgeWeight  *outgoing_weights;
    
    // int *incoming_ptrs;
    // int *incoming_ids;
    // int *incoming_weights;
    
    int *vectorised_outgoing_ids;
    // int *vectorised_incoming_ids;
    
    #ifdef __USE_GPU__
    int gpu_grid_threshold_vertex;
    int gpu_block_threshold_vertex;
    int gpu_warp_threshold_vertex;
    #endif
    
    int *incoming_degrees;
    
    void alloc(int _vertices_count, long long _edges_count);
    void free();
    
    void calculate_incoming_degrees();
    
    void construct_vector_extension();
    
    #ifdef __USE_GPU__
    void estimate_gpu_thresholds();
    #endif
public:
    ExtendedCSRGraph(int _vertices_count = 1, long long _edges_count = 1);
    ~ExtendedCSRGraph();
    
    void resize(int _vertices_count, long long _edges_count);
    
    void print();
    void print_stats() {};
    
    void save_to_graphviz_file(string _file_name, VisualisationMode _visualisation_mode = VISUALISE_AS_DIRECTED);
    bool save_to_binary_file(string file_name);
    bool load_from_binary_file(string file_name);
    
    void import_graph(EdgesListGraph<_TVertexValue, _TEdgeWeight> &_copy_graph, VerticesState _vertices_state,
                      EdgesState _edges_state, int _supported_vector_length,
                      SupportedTraversalType _traversal_type = PULL_TRAVERSAL);
    
    inline int           *get_reordered_vertex_ids() {return reordered_vertex_ids;};
    inline long long     *get_outgoing_ptrs()        {return outgoing_ptrs;};
    inline int           *get_outgoing_ids()         {return outgoing_ids;};
    inline _TEdgeWeight  *get_outgoing_weights()     {return outgoing_weights;};
    inline int           *get_incoming_degrees()     {return incoming_degrees;};
    inline int           *get_vectorised_outgoing_ids() {return vectorised_outgoing_ids;};
    
    #ifdef __USE_GPU__
    void move_to_device();
    void move_to_host();
    #endif
    
    #ifdef __USE_GPU__
    inline int get_gpu_grid_threshold_vertex(){return gpu_grid_threshold_vertex;};
    inline int get_gpu_block_threshold_vertex(){return gpu_block_threshold_vertex;};
    inline int get_gpu_warp_threshold_vertex(){return gpu_warp_threshold_vertex;};
    #endif
    
    void set_threads_count(int _threads_count);
    
    // programing vertices API
    template <class _T> _T*  vertex_array_alloc          ();
    template <class _T> void vertex_array_copy           (_T *_dst_array, _T *_src_array);
    template <class _T> void vertex_array_set_to_constant(_T *_dst_array, _T _value);
    template <class _T> void vertex_array_set_element    (_T *_dst_array, int _pos, _T _value);
    
    // programing edges API
    template <class _T> _T*  allocate_private_caches     (int _threads_count);
    template <class _T> void free_data                   (_T *_array);
    
    // cached API
    template <class _T> inline _T load_vertex_data_cached(int _idx, _T *_data, _T *_private_data);
    template <class _T> inline _T load_vertex_data(int _idx, _T *_data);
    template <class _T> inline _T place_data_into_cache(_T *_data, _T *_private_data);
    template <class _T> inline _T* get_private_data_pointer(_T *_cached_data);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define LOAD_EXTENDED_CSR_GRAPH_DATA(input_graph)                        \
int vertices_count                   = input_graph.get_vertices_count(); \
long long int edges_count            = input_graph.get_edges_count   (); \
\
long long    *outgoing_ptrs           = _graph.get_outgoing_ptrs   ();\
int          *outgoing_ids            = _graph.get_outgoing_ids    ();\
_TEdgeWeight *outgoing_weights        = _graph.get_outgoing_weights();\
int          *vectorised_outgoing_ids = _graph.get_vectorised_outgoing_ids();\

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "extended_CSR_graph.hpp"
#include "programming_API.hpp"
#include "init_graph_helpers.hpp"
#include "gpu_api.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* extended_CSR_graph_h */
