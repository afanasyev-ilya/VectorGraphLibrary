#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class CSR_VG_Graph;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct CSRVertexGroup
{
private:
    int *ids;
    int size;
    int max_size;
    long long neighbours;

    int min_connections, max_connections;
public:
    CSRVertexGroup();
    ~CSRVertexGroup();

    /* print API */
    void print();

    /* get API */
    int *get_ids() {return ids;};
    int get_size() {return size;};
    int get_max_size() {return max_size;};
    long long get_neighbours() {return neighbours;};

    int get_min_connections() {return min_connections;};
    int get_max_connections() {return max_connections;};

    bool id_in_range(int _src_id, int _connections_count);

    /* modification API */
    void import(CSR_VG_Graph *_graph, int _bottom, int _top);
    void copy(CSRVertexGroup &_other_group);
    void add_vertex(int _src_id);
    void clear() {size = 0;};

    template <typename CopyCond>
    void copy_data_if(CSRVertexGroup &_full_group, CopyCond copy_cond, int *_buffer);

    void resize(int _new_size);

    #ifdef __USE_GPU__
    void move_to_host();
    #endif

    #ifdef __USE_GPU__
    void move_to_device();
    #endif
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define LOAD_CSR_VERTEX_GROUP_DATA(vertex_group)  \
int *ids = vertex_group.get_ids(); \
int size = vertex_group.get_size(); \
int max_size = vertex_group.get_max_size(); \
long long neighbours = vertex_group.get_neighbours(); \

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
