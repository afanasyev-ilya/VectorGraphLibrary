#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
#define any_arch_func __host__ __device__
#else
#define any_arch_func
#endif

class UndirectedGraph : public BaseGraph
{
protected:
    bool is_copy;
public:
    UndirectedGraph() {};
    ~UndirectedGraph() {};

    /* get/set API */

    virtual any_arch_func int get_connections_count(int _vertex_id) = 0;
    virtual any_arch_func int get_edge_dst(int _src_id, int _edge_pos) = 0;

    virtual size_t get_edges_array_index(int _v, int _edge_pos) = 0;
    virtual size_t get_edges_array_direction_shift_size() = 0;

    /* reorder API */

    virtual int reorder_to_sorted(int _vertex_id) = 0;
    virtual int reorder_to_original(int _vertex_id) = 0;

    inline int select_random_vertex() { return rand() % this->vertices_count; };
    virtual int select_random_nz_vertex() = 0;

    virtual void reorder_to_original(char *_data, char *_buffer, size_t _elem_size) = 0;
    virtual void reorder_to_sorted(char *_data, char *_buffer, size_t _elem_size) = 0;

    virtual void reorder_edges_gather(char *_src, char *_dst, size_t _elem_size) = 0;
    virtual void reorder_edges_scatter(char *_src, char *_dst, size_t _elem_size) = 0;

    /* file load/store API */
    virtual void save_main_content_to_binary_file(FILE *_graph_file) = 0;
    virtual void load_main_content_from_binary_file(FILE *_graph_file) = 0;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

