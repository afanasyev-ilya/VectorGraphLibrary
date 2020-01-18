//
//  vectorised_CSR_graph.h
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 19/04/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef vectorised_CSR_graph_h
#define vectorised_CSR_graph_h

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <vector>
#include <algorithm>
#include <string>
#include <fstream>
#include <stdio.h>
#include <float.h>

#include "../common/vectorise_CSR.h"

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
class VectorisedCSRGraph : public BaseGraph<_TVertexValue, _TEdgeWeight>
{
private:
    VerticesState vertices_state;
    EdgesState edges_state;
    int supported_vector_length;
    
    int vector_segments_count;
    int number_of_vertices_in_first_part;
    int vertices_in_vector_segments;
    
    // data to restore old graph
    int *reordered_vertex_ids;
    int old_vertices_count;
    
    // vector segments data
    long long *vector_group_ptrs;
    int *vector_group_sizes;
    
    // first part data
    long long *first_part_ptrs;
    int *first_part_sizes;
    
    // outgoing edges data
    int *outgoing_ids;
    _TEdgeWeight *outgoing_weights;
    
    // additional data structure for page rank
    int *incoming_sizes_per_vertex;
    
    // number of threads used in parallel regions
    int threads_count;
    
    void alloc(int _vertices_count, long long _edges_count, int _vertices_in_first_part = 0);
    void free();
    
    #ifdef __USE_GPU__
    int gpu_grid_threshold_vertex;
    int gpu_block_threshold_vertex;
    int gpu_warp_threshold_vertex;
    #endif
    
    // helper functions of init graph function
    void construct_tmp_graph_from_edges_list(EdgesListGraph<_TVertexValue, _TEdgeWeight> &_old_graph,
                                             vector<vector<TempEdgeData<_TEdgeWeight> > > &_tmp_graph,
                                             vector<_TVertexValue> &_tmp_vertex_values,
                                             int _tmp_vertices_count,
                                             SupportedTraversalType _traversal_type);
    
    void sort_vertices_in_descending_order(vector<vector<TempEdgeData<_TEdgeWeight> > > &_tmp_graph,
                                           int *_tmp_reordered_vertex_ids,
                                           int _tmp_vertices_count);
    
    void sort_edges(vector<vector<TempEdgeData<_TEdgeWeight> > > &_tmp_graph, int _tmp_vertices_count);
    
    int calculate_and_find_threshold_vertex(vector<vector<TempEdgeData<_TEdgeWeight> > > &_tmp_graph,
                                            int _tmp_vertices_count, long long _tmp_edges_count);
    
    void flatten_graph(vector<vector<TempEdgeData<_TEdgeWeight> > > &_tmp_graph, int &_tmp_vertices_count,
                       long long int &_tmp_edges_count, long long int _old_edges_count, int _last_part_start);
    
    void convert_tmp_graph_into_vect_CSR(vector<vector<TempEdgeData<_TEdgeWeight> > > &_tmp_graph,
                                         int *_tmp_reordered_vertex_ids, vector<_TVertexValue> &_tmp_vertex_values,
                                         int _tmp_vertices_count);
    
    void calculate_incoming_sizes();
    
    #ifdef __USE_GPU__
    void estimate_gpu_thresholds();
    #endif
public:
    VectorisedCSRGraph(int _vertices_count = VECTOR_LENGTH, long long _edges_count = 1);
    ~VectorisedCSRGraph();
    
    void resize(int _vertices_count, long long _edges_count, int _vertices_in_first_part = 0);
    
    void print();
    void print_stats() {};
    
    void save_to_graphviz_file(string _file_name, VisualisationMode _visualisation_mode = VISUALISE_AS_DIRECTED);
    bool save_to_binary_file(string file_name);
    bool load_from_binary_file(string file_name);
    
    void import_graph(EdgesListGraph<_TVertexValue, _TEdgeWeight> &_old_graph, VerticesState _vertices_state,
                      EdgesState _edges_state, int _supported_vector_length,
                      SupportedTraversalType _traversal_type = PULL_TRAVERSAL,
                      bool _free_initial_graph = false);
    
    inline int get_number_of_vertices_in_first_part() {return number_of_vertices_in_first_part;};
    inline int get_vector_segments_count() {return vector_segments_count;};
    
    #ifdef __USE_GPU__
    inline int get_gpu_grid_threshold_vertex(){return gpu_grid_threshold_vertex;};
    inline int get_gpu_block_threshold_vertex(){return gpu_block_threshold_vertex;};
    inline int get_gpu_warp_threshold_vertex(){return gpu_warp_threshold_vertex;};
    #endif
    
    inline int           *get_reordered_vertex_ids     () {return reordered_vertex_ids;};
    inline long long     *get_first_part_ptrs          () {return first_part_ptrs;};
    inline int           *get_first_part_sizes         () {return first_part_sizes;};
    inline long long     *get_vector_group_ptrs        () {return vector_group_ptrs;};
    inline int           *get_vector_group_sizes       () {return vector_group_sizes;};
    inline int           *get_outgoing_ids             () {return outgoing_ids;};
    inline _TEdgeWeight  *get_outgoing_weights         () {return outgoing_weights;};
    inline int           *get_incoming_sizes_per_vertex() {return incoming_sizes_per_vertex;};
    
    #ifdef __USE_GPU__
    void move_to_device();
    void move_to_host();
    #endif
    
    void set_threads_count(int _threads_count);
    
    // programing vertices API
    template <class _T> _T*  vertex_array_alloc          ();
    template <class _T> void vertex_array_copy           (_T *_dst_array, _T *_src_array);
    template <class _T> void vertex_array_set_to_constant(_T *_dst_array, _T _value);
    template <class _T> void vertex_array_set_element    (_T *_dst_array, int _pos, _T _value);
    
    // programing edges API
    template <class _T> _T*  edges_array_alloc           ();
    template <class _T> void gather_all_edges_data       (_T *_dst_array, _T *_src_array);
    template <class _T> void gather_all_edges_data_cached(_T *_dst_array, _T *_src_array, _T *_cached_src_array);
    template <class _T> _T*  allocate_private_caches     (int _threads_count);
    template <class _T> void free_data                   (_T *_array);
    
    template <class _T> void gather_all_edges_data_intervals_only(_T *_dst_array, _T *_src_array, _T *_cached_src_array);
    template <class _T> void gather_all_edges_data_threads_only(_T *_dst_array, _T *_src_array, _T *_cached_src_array);
    
    // vertex traversal API
    long long get_vertex_pointer(int src_id);
    int get_vector_connections_count(int src_id);
    
    // cached API
    template <class _T> inline _T load_vertex_data_cached(int _idx, _T *_data, _T *_private_data);
    template <class _T> inline _T load_vertex_data(int _idx, _T *_data);
    template <class _T> inline _T place_data_into_cache(_T *_data, _T *_private_data);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "vectorised_CSR_graph.hpp"
#include "init_graph_helpers.hpp"
#include "programming_API.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define LOAD_VECTORISED_CSR_GRAPH_REVERSE_DATA(input_graph)                                \
int vertices_count                   = input_graph.get_vertices_count                  (); \
long long int edges_count            = input_graph.get_edges_count                     (); \
int vector_segments_count            = input_graph.get_vector_segments_count           (); \
int number_of_vertices_in_first_part = input_graph.get_number_of_vertices_in_first_part(); \
                                                                                           \
int           *reordered_vertex_ids = input_graph.get_reordered_vertex_ids     (); \
long long     *first_part_ptrs      = input_graph.get_first_part_ptrs          (); \
int           *first_part_sizes     = input_graph.get_first_part_sizes         (); \
long long     *vector_group_ptrs    = input_graph.get_vector_group_ptrs        (); \
int           *vector_group_sizes   = input_graph.get_vector_group_sizes       (); \
int           *incoming_ids         = input_graph.get_outgoing_ids             (); \
_TEdgeWeight  *incoming_weights     = input_graph.get_outgoing_weights         (); \
int *outgoing_sizes                 = input_graph.get_incoming_sizes_per_vertex(); \

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define LOAD_VECTORISED_CSR_GRAPH_DATA(input_graph)                                        \
int vertices_count                   = input_graph.get_vertices_count                  (); \
long long int edges_count            = input_graph.get_edges_count                     (); \
int vector_segments_count            = input_graph.get_vector_segments_count           (); \
int number_of_vertices_in_first_part = input_graph.get_number_of_vertices_in_first_part(); \
\
int           *reordered_vertex_ids = input_graph.get_reordered_vertex_ids     (); \
long long     *first_part_ptrs      = input_graph.get_first_part_ptrs          (); \
int           *first_part_sizes     = input_graph.get_first_part_sizes         (); \
long long     *vector_group_ptrs    = input_graph.get_vector_group_ptrs        (); \
int           *vector_group_sizes   = input_graph.get_vector_group_sizes       (); \
int           *outgoing_ids         = input_graph.get_outgoing_ids             (); \
_TEdgeWeight  *outgoing_weights     = input_graph.get_outgoing_weights         (); \
int *outgoing_sizes                 = input_graph.get_incoming_sizes_per_vertex(); \

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* vectorised_CSR_graph_h */
