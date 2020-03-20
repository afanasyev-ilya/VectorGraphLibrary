/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class GraphAnalytics
{
private:
    pair<int, int> calculate_power_range(int _val);
    map<int, int> calculate_degree_distribution(long long *_adjacent_ptrs, int _vertices_count);

    template <typename _TVertexValue, typename _TEdgeWeight>
    void print_graph_memory_consumption(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph);

    void analyse_component_stats(int *_components, int _vertices_count);

    template <typename _TVertexValue, typename _TEdgeWeight>
    void analyse_graph_thresholds(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph);
public:

    template <typename _TVertexValue, typename _TEdgeWeight>
    void analyse_graph_stats(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, string _graph_name);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_analytics.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
