/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class GraphAnalytics
{
private:
    pair<long long, long long> calculate_power_range(long long _val);
    map<int, int> calculate_degree_distribution(long long *_adjacent_ptrs, int _vertices_count);


    void print_graph_memory_consumption(VectCSRGraph &_graph);

    void analyse_component_stats(int *_components, int _vertices_count);


    void analyse_graph_thresholds(VectCSRGraph &_graph);
public:
    void analyse_graph_stats(VectCSRGraph &_graph, string _graph_name);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_analytics.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
