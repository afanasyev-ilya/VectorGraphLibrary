#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class CSR_VG_Graph;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct CSRVertexGroup
{
    int *ids;
    int size;
    int max_size;
    long long neighbours;

    int min_connections, max_connections;

    CSRVertexGroup();
    ~CSRVertexGroup();

    void import(CSR_VG_Graph *_graph, int _bottom, int _top);

    void copy(CSRVertexGroup &_other_group);

    bool id_in_range(int _src_id, int _connections_count);

    void add_vertex(int _src_id);

    template <typename CopyCond>
    void copy_data_if(CSRVertexGroup &_full_group, CopyCond copy_cond, int *_buffer);

    void resize(int _new_size);

    void print();

    #ifdef __USE_GPU__
    void move_to_host();
    #endif

    #ifdef __USE_GPU__
    void move_to_device();
    #endif
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
