#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class UndirectedGraph : public BaseGraph
{
protected:

public:
    UndirectedGraph() {};
    ~UndirectedGraph() {};

    virtual int get_connections_count(int _vertex_id) = 0;
    virtual int get_edge_dst(int _src_id, int _edge_pos) = 0;

    virtual int reorder_to_sorted(int _vertex_id) = 0;
    virtual int reorder_to_original(int _vertex_id) = 0;

    inline int select_random_vertex() { return rand() % this->vertices_count; };
    virtual int select_random_nz_vertex() = 0;

    virtual void reorder_to_original(char *_data, char *_buffer, size_t _elem_size) = 0;
    virtual void reorder_to_sorted(char *_data, char *_buffer, size_t _elem_size) = 0;

    /* file load/store API */
    virtual void save_main_content_to_binary_file(FILE *_graph_file) = 0;
    virtual void load_main_content_from_binary_file(FILE *_graph_file) = 0;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
