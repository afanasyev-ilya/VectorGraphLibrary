#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class BaseFrontier
{
protected:
    FrontierClassType class_type;

    int size;
    VGL_Graph *graph_ptr;
    TraversalDirection direction;

    int neighbours_count;

    // all frontiers have flags and ids - ?
    int *flags;
    int *ids;
    int *work_buffer;
    FrontierSparsityType sparsity_type;
public:
    BaseFrontier(VGL_Graph &_graph, TraversalDirection _direction)
    {
        graph_ptr = &_graph; direction = _direction; class_type = BASE_FRONTIER;
    };

    // get API
    inline int get_size() { return size; };
    inline int *get_ids() { return ids; };
    inline int *get_flags() { return flags; };
    inline int *get_work_buffer() { return work_buffer; };
    inline FrontierSparsityType get_sparsity_type() { return sparsity_type; };
    inline FrontierClassType get_class_type() { return class_type; };
    inline long long get_neighbours_count() { return neighbours_count; };

    // printing API
    virtual void print_stats() = 0;
    virtual void print() = 0;

    // frontier modification API
    virtual void add_vertex(int _src_id) = 0;
    virtual void add_group_of_vertices(int *_vertex_ids, int _number_of_vertices) = 0;
    virtual void clear() = 0;

    // modification
    virtual void set_all_active() = 0;

    // frontier direction API
    TraversalDirection get_direction() { return direction; };
    void set_direction(TraversalDirection _direction) { direction = _direction; };
    void reorder(TraversalDirection _direction) { set_direction(_direction); };

    #ifdef __USE_GPU__
    void move_to_host();
    void move_to_device();
    #endif
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
void BaseFrontier::move_to_host()
{
    int vertices_count = graph_ptr->get_vertices_count();
    MemoryAPI::move_array_to_host(flags, vertices_count);
    MemoryAPI::move_array_to_host(ids, vertices_count);
    MemoryAPI::move_array_to_host(work_buffer, vertices_count);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
void BaseFrontier::move_to_device()
{
    int vertices_count = graph_ptr->get_vertices_count();
    MemoryAPI::move_array_to_device(flags, vertices_count);
    MemoryAPI::move_array_to_device(ids, vertices_count);
    MemoryAPI::move_array_to_device(work_buffer, vertices_count);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define LOAD_FRONTIER_DATA(frontier)              \
int frontier_size          = frontier.get_size(); \
long long frontier_neighbours_count = frontier.get_neighbours_count(); \
int *frontier_ids          = frontier.get_ids(); \
int *frontier_flags        = frontier.get_flags();   \
int *frontier_work_buffer  = frontier.get_work_buffer();  \

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
