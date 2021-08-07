#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <vector>
#include <algorithm>
#include <string>
#include <fstream>
#include <stdio.h>
#include <string.h>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
class VerticesArray;

template <typename _T>
class EdgesArray;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class CSRGraph: public UndirectedGraph
{
private:
    long long     *vertex_pointers;
    int           *adjacent_ids;

    vgl_sort_indexes *edges_reorder_indexes; // allows to convert VectorCSRGraph edges (and weights) from sorted to original order

    void alloc(int _vertices_count, long long _edges_count);
    void free();

    /* dummy reorder API */
    void reorder_to_sorted(char *_data, char *_buffer, size_t _elem_size) {return;};
    void reorder_to_original(char *_data, char *_buffer, size_t _elem_size) {return;};
    int reorder_to_sorted(int _vertex_id) { return _vertex_id; };
    int reorder_to_original(int _vertex_id) { return _vertex_id; };

    void construct_unsorted_csr(EdgesContainer &_edges_container);
    void copy_edges_indexes(vgl_sort_indexes *_sort_indexes);

    /* file load/store API */
    void save_main_content_to_binary_file(FILE *_graph_file) final {};
    void load_main_content_from_binary_file(FILE *_graph_file) final {};
public:
    CSRGraph(int _vertices_count = 1, long long _edges_count = 1);
    ~CSRGraph();

    /* get API */
    inline long long *get_vertex_pointers() {return vertex_pointers;};
    inline int       *get_adjacent_ids()    {return adjacent_ids;};

    inline int get_connections_count(int _src_id) final {return vertex_pointers[_src_id+1] - vertex_pointers[_src_id];};
    inline int get_edge_dst(int _src_id, int _edge_pos) final {return adjacent_ids[vertex_pointers[_src_id] + _edge_pos];};

    /* print API */
    void print();
    void print_size();
    void print_stats() {};
    size_t get_size();

    /* file load/store API */
    bool save_to_binary_file(string _file_name);
    bool load_from_binary_file(string _file_name);

    // resize graph
    void resize(int _vertices_count, long long _edges_count);

    /* import and preprocess API */
    // creates VectorCSRGraph format from EdgesListGraph
    void import(EdgesContainer &_edges_container);

    template <typename _T>
    void reorder_edges_gather(_T *_src, _T *_dst)
    {
        #if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
        #pragma omp parallel
        {
            openmp_reorder_wrapper_gather_inplace(_src, _dst, edges_reorder_indexes, this->edges_count);
        }
        #endif

        #if defined(__USE_GPU__)
        cuda_reorder_wrapper_gather_inplace(_src, _dst, edges_reorder_indexes, this->edges_count);
        #endif
    }

    template <typename _T>
    void reorder_edges_gather(_T *_src, _T *_dst)
    {
        #if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
        #pragma omp parallel
        {
            openmp_reorder_scatter_gather_inplace(_src, _dst, edges_reorder_indexes, this->edges_count);
        }
        #endif

        #if defined(__USE_GPU__)
        cuda_reorder_wrapper_scatter_inplace(_src, _dst, edges_reorder_indexes, this->edges_count);
        #endif
    }

    /* vertices API */
    int select_random_nz_vertex();

    friend class GraphAbstractions;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define LOAD_CSR_GRAPH_DATA(input_graph)  \
int vertices_count            = input_graph.get_vertices_count(); \
long long int edges_count     = input_graph.get_edges_count   (); \
\
long long    *vertex_pointers = input_graph.get_vertex_pointers   ();\
int          *adjacent_ids    = input_graph.get_adjacent_ids    ();\

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "csr_graph.hpp"
#include "import.hpp"
#include "print.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

