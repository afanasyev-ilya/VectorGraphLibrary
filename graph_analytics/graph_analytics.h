/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class GraphAnalytics
{
private:
    static pair<long long, long long> calculate_power_range(long long _val);
    static map<int, int> calculate_degree_distribution(long long *_adjacent_ptrs, int _vertices_count);

    static void print_graph_memory_consumption(VectCSRGraph &_graph);

    static void analyse_component_stats(int *_components, int _vertices_count);

    static void analyse_graph_thresholds(VectCSRGraph &_graph);
public:
    static void analyse_degrees(VectorCSRGraph &_graph);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
