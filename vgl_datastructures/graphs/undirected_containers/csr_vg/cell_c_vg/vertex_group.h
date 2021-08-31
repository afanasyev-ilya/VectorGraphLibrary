#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class CSR_VG_Graph;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct CSRVertexGroupCellC
{
    int *ids;
    int size;
    int max_size;
    long long neighbours;

    int min_connections, max_connections;

    CSRVertexGroupCellC();
    ~CSRVertexGroupCellC();

    void import(CSR_VG_Graph *_graph, int _bottom, int _top);

    void copy(CSRVertexGroupCellC &_other_group);

    bool id_in_range(int _src_id, int _connections_count);

    void add_vertex(int _src_id);

    template <typename CopyCond>
    void copy_data_if(CSRVertexGroupCellC &_full_group, CopyCond copy_cond, int *_buffer);

    void resize(int _new_size);

    void print_ids();

    #ifdef __USE_GPU__
    void move_to_host();
    #endif

    #ifdef __USE_GPU__
    void move_to_device();
    #endif
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
