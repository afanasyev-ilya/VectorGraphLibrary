#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <math.h>
#include <sstream>
#include <string>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum ConvertDirectionType
{
    DirectedToDirected = 0,
    DirectedToUndirected = 1,
    UndirectedToDirected = 2,
    UndirectedToUndirected = 3
};

enum DirectionType
{
    UNDIRECTED_GRAPH = 0,
    DIRECTED_GRAPH = 1
};

class EdgesContainer
{
private:
    int vertices_count;
    long long edges_count;
    int *src_ids;
    int *dst_ids;
public:
    int *get_src_ids() {return src_ids;};
    int *get_dst_ids() {return dst_ids;};
    int get_vertices_count() {return vertices_count;};
    long long get_edges_count() {return edges_count;};

    void transpose()
    {
        int *tmp = src_ids;
        src_ids = dst_ids;
        dst_ids = tmp;
    }

    void resize(int _vertices_count, long long _edges_count)
    {
        vertices_count = _vertices_count;
        edges_count = _edges_count;

        MemoryAPI::free_array(src_ids);
        MemoryAPI::free_array(dst_ids);
        MemoryAPI::allocate_array(&src_ids, edges_count);
        MemoryAPI::allocate_array(&dst_ids, edges_count);
    }

    EdgesContainer(int _vertices_count = 1, long long _edges_count = 1)
    {
        vertices_count = _vertices_count;
        edges_count = _edges_count;
        MemoryAPI::allocate_array(&src_ids, edges_count);
        MemoryAPI::allocate_array(&dst_ids, edges_count);
    }

    ~EdgesContainer()
    {
        MemoryAPI::free_array(src_ids);
        MemoryAPI::free_array(dst_ids);
    }

    void preprocess_into_csr_based(int *_work_buffer, vgl_sort_indexes *_sort_buffer)
    {
        bool work_buffer_was_allocated = false;
        if(_work_buffer == NULL)
        {
            work_buffer_was_allocated = true;
            MemoryAPI::allocate_array(&_work_buffer, this->edges_count);
        }

        bool sort_buffer_was_allocated = false;
        if(_sort_buffer == NULL)
        {
            sort_buffer_was_allocated = true;
            MemoryAPI::allocate_array(&_sort_buffer, this->edges_count);
        }

        // init sort indexes
        #pragma _NEC ivdep
        #pragma omp parallel for
        for(int i = 0; i < this->edges_count; i++)
        {
            _sort_buffer[i] = i;
        }

        // sort src_ids
        Timer tm;
        tm.start();
        Sorter::sort(src_ids, _sort_buffer, this->edges_count, SORT_ASCENDING);
        tm.end();
        #ifdef __PRINT_API_PERFORMANCE_STATS__
        tm.print_time_stats("EdgesListGraph sorting (to CSR) time");
        #endif

        // reorder dst_ids
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vector
        #pragma _NEC gather_reorder
        #pragma omp parallel for
        for(long long edge_pos = 0; edge_pos < this->edges_count; edge_pos++)
        {
            _work_buffer[edge_pos] = dst_ids[_sort_buffer[edge_pos]];
        }

        #pragma _NEC ivdep
        #pragma omp parallel for
        for(long long edge_pos = 0; edge_pos < this->edges_count; edge_pos++)
        {
            dst_ids[edge_pos] = _work_buffer[edge_pos];
        }

        if(work_buffer_was_allocated)
        {
            MemoryAPI::free_array(_work_buffer);
        }
        if(sort_buffer_was_allocated)
        {
            MemoryAPI::free_array(_sort_buffer);
        }
    }

    void renumber_vertices(int *_conversion_array, int *_work_buffer)
    {
        Timer tm;
        tm.start();

        bool work_buffer_was_allocated = false;
        if(_work_buffer == NULL)
        {
            work_buffer_was_allocated = true;
            MemoryAPI::allocate_array(&_work_buffer, this->edges_count);
        }

        #pragma _NEC ivdep
        #pragma _NEC novob
        #pragma _NEC vector
        #pragma _NEC gather_reorder
        #pragma omp parallel for
        for(long long edge_pos = 0; edge_pos < this->edges_count; edge_pos++)
        {
            _work_buffer[edge_pos] = _conversion_array[src_ids[edge_pos]];
        }

        #pragma _NEC ivdep
        #pragma omp parallel for
        for(long long edge_pos = 0; edge_pos < this->edges_count; edge_pos++)
        {
            src_ids[edge_pos] = _work_buffer[edge_pos];
        }

        #pragma _NEC ivdep
        #pragma _NEC novob
        #pragma _NEC vector
        #pragma _NEC gather_reorder
        #pragma omp parallel for
        for(long long edge_pos = 0; edge_pos < this->edges_count; edge_pos++)
        {
            _work_buffer[edge_pos] = _conversion_array[dst_ids[edge_pos]];
        }

        #pragma _NEC ivdep
        #pragma omp parallel for
        for(long long edge_pos = 0; edge_pos < this->edges_count; edge_pos++)
        {
            dst_ids[edge_pos] = _work_buffer[edge_pos];
        }

        if(work_buffer_was_allocated)
        {
            MemoryAPI::free_array(_work_buffer);
        }

        tm.end();
        #ifdef __PRINT_API_PERFORMANCE_STATS__
        tm.print_time_and_bandwidth_stats("EdgesList graph reorder (to optimized)", this->edges_count, sizeof(int)*(2*2 + 3*2));
        #endif
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class GraphGenerationAPI
{
public:
    static void random_uniform(EdgesContainer &_edges_container,
                               int _vertices_count,
                               long long _edges_count,
                               DirectionType _direction_type = DIRECTED_GRAPH);
    
    static void R_MAT(EdgesContainer &_edges_container,
                      int _vertices_count,
                      long long _edges_count,
                      int _a_prob,
                      int _b_prob,
                      int _c_prob,
                      int _d_prob,
                      DirectionType _direction_type = DIRECTED_GRAPH);
    
    /*static void SSCA2(EdgesListGraph &_graph,
                      int _vertices_count,
                      int _max_clique_size);
    
    static void SCC_uniform(EdgesListGraph &_graph,
                            int _vertices_count,
                            int _min_scc_size,
                            int _max_scc_size);
    
    static void init_from_txt_file(EdgesListGraph &_graph,
                                   string _txt_file_name,
                                   DirectionType _direction_type = DIRECTED_GRAPH);*/
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_generation_API.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
