#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class BaseGraph
{
protected:
    GraphType graph_type;
    
    int vertices_count;
    long long edges_count;
    
    bool graph_on_device;

    SupportedDirection supported_direction;
    bool can_use_gather();
    bool can_use_scatter();
public:
    BaseGraph() {this->graph_on_device = false;};
    ~BaseGraph() {};

    /* get API */
    inline int get_vertices_count() {return vertices_count;};
    inline long long get_edges_count() {return edges_count;};
    inline GraphType get_type() {return graph_type;};

    /* print API */
    virtual void print() = 0;
    virtual void print_size() = 0;
    virtual size_t get_size() = 0;

    /* file load/store API */
    virtual bool save_to_binary_file(string _file_name) = 0;
    virtual bool load_from_binary_file(string _file_name) = 0;

    /* GPU specific (copy) API */
    #ifdef __USE_GPU__
    virtual void move_to_device() = 0;
    virtual void move_to_host() = 0;
    #endif
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool BaseGraph::can_use_gather()
{
    if((supported_direction == USE_BOTH) || (supported_direction == USE_GATHER_ONLY))
        return true;
    else
        return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool BaseGraph::can_use_scatter()
{
    if((supported_direction == USE_BOTH) || (supported_direction == USE_SCATTER_ONLY))
        return true;
    else
        return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

