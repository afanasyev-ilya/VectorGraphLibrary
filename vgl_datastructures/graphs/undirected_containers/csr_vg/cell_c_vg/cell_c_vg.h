#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class CSR_VG_Graph;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct CSRVertexGroupCellC
{
    int *ids;
    int size;

    // ve repr start
    int vector_segments_count;
    long long edges_count_in_ve;

    long long *vector_group_ptrs;
    int *vector_group_sizes;
    int *adjacent_ids;
    // ve repr end

    int min_connections, max_connections;

    CSRVertexGroupCellC();
    ~CSRVertexGroupCellC();

    void import(CSR_VG_Graph *_graph, int _bottom, int _top);

    bool id_in_range(int _src_id, int _connections_count);

    void add_vertex(int _src_id);

    void resize(int _new_size);

    void print_ids();

    #ifdef __USE_GPU__
    void move_to_host() {};
    #endif

    #ifdef __USE_GPU__
    void move_to_device() {};
    #endif
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
