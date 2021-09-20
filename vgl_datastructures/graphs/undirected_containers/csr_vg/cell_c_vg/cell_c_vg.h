#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class CSR_VG_Graph;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct CSRVertexGroupCellC
{
private:
    int *vertex_ids; // ids of vertices from this group
    int size; // size of this group

    int vector_segments_count; // similar to VE implementation
    long long edges_count_in_ve;

    long long *vector_group_ptrs;
    int *vector_group_sizes;
    int *vector_group_adjacent_ids;

    long long *old_edge_indexes; // edge IDS required for conversion (take + O|V| space)

    int min_connections, max_connections;

    // helper functions
    bool id_in_range(int _src_id, int _connections_count);
public:
    // init functions
    CSRVertexGroupCellC();
    ~CSRVertexGroupCellC();

    /* print API */
    void print();

    /* get API */
    int *get_vertex_ids() {return vertex_ids;};
    int get_size() {return size;};

    int get_vector_segments_count() {return vector_segments_count;};
    long long get_edges_count_in_ve() {return edges_count_in_ve;};

    long long *get_vector_group_ptrs() {return vector_group_ptrs;};
    int *get_vector_group_sizes() {return vector_group_sizes;};
    int *get_vector_group_adjacent_ids() {return vector_group_adjacent_ids;};

    long long *get_old_edge_indexes() {return old_edge_indexes;};

    int get_min_connections() {return min_connections;};
    int get_max_connections() {return max_connections;};

    /* import and preprocess API */
    void import(CSR_VG_Graph *_graph, int _bottom, int _top);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
