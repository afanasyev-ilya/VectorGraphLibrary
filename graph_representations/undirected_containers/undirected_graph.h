#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class UndirectedGraph : public BaseGraph
{
protected:

public:
    UndirectedGraph() {};
    ~UndirectedGraph() {};

    virtual int get_connections_count(int _vertex_id) = 0;

    virtual int reorder_to_sorted(int _vertex_id) = 0;
    virtual int reorder_to_original(int _vertex_id) = 0;

    virtual int select_random_vertex() = 0;

    // get vertex();
    // get adjacent edges - bad
    //
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

