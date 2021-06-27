#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void GraphAnalytics::analyse_component_stats(int *_components, int _vertices_count)
{
    map<int, int> component_sizes;
    for(int i = 0; i < _vertices_count; i++)
    {
        int current_component = _components[i];
        component_sizes[current_component]++;
    }

    map<int, int> sizes_stats;

    cout << " -------------------------------- CC stats --------------------------------- " << endl;
    for(auto it = component_sizes.begin(); it != component_sizes.end(); it++)
    {
        sizes_stats[it->second]++;
    }

    std::vector<std::pair<int, int>> size_pairs;
    for (auto itr = sizes_stats.begin(); itr != sizes_stats.end(); ++itr)
        size_pairs.push_back(*itr);

    auto cmp_func = [=](const std::pair<int, int>& a, const std::pair<int, int>& b) { return a.first >= b.first; };

    sort(size_pairs.begin(), size_pairs.end(), cmp_func);

    int larges_component = 0;
    long long non_trivial_components_count = 0;
    for(auto it = size_pairs.begin(); it != size_pairs.end(); it++)
    {
        cout << "there are " << it->second << " components of size " << it->first << endl;
        if(it->first > larges_component)
            larges_component = it->first;
        if(it->first > 1)
            non_trivial_components_count += it->second;
    }
    cout << endl;
    cout << "largest component size: " << larges_component << ", " << 100.0 * larges_component/_vertices_count << " % of all vertices" << endl;
    cout << "number of non-trivial components: " << non_trivial_components_count << endl;
    cout << " ---------------------------------------------------------------------------- " << endl << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

pair<long long, long long> GraphAnalytics::calculate_power_range(long long _val)
{
    double val = _val;
    long long low_border = (long long)(log2(_val));
    long long high_border = (long long)(log2(_val)) + 1;

    return (pair<long long, long long>(low_border, high_border));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

map<int, int> GraphAnalytics::calculate_degree_distribution(long long *_vertex_pointers, int _vertices_count)
{
    map<int, int> degree_distribution;

    for(int v = 0; v < _vertices_count; v++)
    {
        int connections_count = _vertex_pointers[v + 1] - _vertex_pointers[v];
        auto range = calculate_power_range(connections_count);

        if(connections_count == 0)
            degree_distribution[-1]++;
        else
            degree_distribution[range.first]++;
    }

    return degree_distribution;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void GraphAnalytics::print_graph_memory_consumption(VectCSRGraph &_graph)
{
    /*LOAD_UNDIRECTED_VECT_CSR_GRAPH_DATA(_graph);

    double no_wights_size_in_bytes = sizeof(_TVertexValue)*vertices_count + sizeof(long long)*(vertices_count + 1) + sizeof(int)*edges_count;
    double weights_size_in_bytes = sizeof(_TEdgeWeight)*edges_count;
    double edges_in_ve = ve_vector_group_ptrs[ve_vector_segments_count - 1] - ve_vector_group_ptrs[0];
    double ve_size_in_bytes = (sizeof(int) + sizeof(_TEdgeWeight))*edges_in_ve;

    cout << "no-weights size (CSR): " << no_wights_size_in_bytes / 1e9 << " GB" << endl;
    cout << "size with weights (CSR): " << (no_wights_size_in_bytes + weights_size_in_bytes) / 1e9 << " GB" << endl;
    cout << "ve size: " << ve_size_in_bytes / 1e9 << " GB" << endl;
    cout << "indirectly accessed array size: " << sizeof(int)*vertices_count / 1e6 << " MB" << endl;*/
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void GraphAnalytics::analyse_graph_thresholds(VectCSRGraph &_graph)
{
    /*#ifdef __USE_NEC_SX_AURORA__
    cout << endl;
    cout << "large interval: (" << 0 << " ," << _graph.get_vector_engine_threshold_vertex() << ")" << endl;
    cout << "medium interval: (" << _graph.get_vector_engine_threshold_vertex() << " ," << _graph.get_vector_core_threshold_vertex() << ")" << endl;
    cout << "small interval: (" << _graph.get_vector_core_threshold_vertex() << " ," << _graph.get_vertices_count() << ")" << endl << endl;
    #endif*/
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void GraphAnalytics::analyse_degrees(UndirectedVectCSRGraph &_graph)
{
    int vertices_count = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count();
    long long *vertex_pointers = _graph.get_vertex_pointers();

    auto vertex_power_range = calculate_power_range(vertices_count);
    cout << "vertices count: " << vertices_count << " | in range: (" << vertex_power_range.first << "," << vertex_power_range.second << ")" << endl;
    cout << "approximate vc: " << vertices_count / 1e6 << " bn" << endl;

    auto edges_power_range = calculate_power_range(edges_count);
    cout << "edges count: " << edges_count << " | in range: (" << edges_power_range.first << "," << edges_power_range.second << ")" << endl;
    cout << "approximate ec: " << edges_count / 1e6 << " bn" << endl;
    cout << "average degree: " << edges_count/vertices_count << endl << endl;

    cout << "highest vertex degree: " << vertex_pointers[1] - vertex_pointers[0] << ", " << 100.0*(vertex_pointers[1] - vertex_pointers[0])/edges_count << " % of all edges" << endl;
    auto degree_distribution = calculate_degree_distribution(vertex_pointers, vertices_count);
    cout << "degree_distribution:" << endl;
    long long zero_nodes_count = 0;
    for(auto it = degree_distribution.rbegin(); it != degree_distribution.rend(); it++)
    {
        if(it->first != -1)
        {
            cout << "(2^" << it->first << ",2^" << it->first + 1 << "): " << it->second << endl;
        }
        else
        {
            zero_nodes_count = it->second;
            cout << "zero-nodes: " << it->second << endl;
        }
    }
    cout << "zero-nodes percent: " << 100*zero_nodes_count/vertices_count << "% from total vertices" << endl;
    cout << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
