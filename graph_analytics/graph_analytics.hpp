#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void GraphAnalytics::analyse_graph_stats(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, string _graph_name)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);
    long long *adjacent_ptrs = outgoing_ptrs;

    cout << "------------------------------------- graph stats -----------------------------------------" << endl;
    auto vertex_power_range = calculate_power_range(vertices_count);
    cout << "vertices count: " << vertices_count << " | in range: (" << vertex_power_range.first << "," << vertex_power_range.second << ")" << endl;

    auto edges_power_range = calculate_power_range(edges_count);
    cout << "edges count: " << edges_count << " | in range: (" << edges_power_range.first << "," << edges_power_range.second << ")" << endl;
    cout << "average degree: " << edges_count/vertices_count << endl << endl;

    cout << "highest vertex degree: " << adjacent_ptrs[1] - adjacent_ptrs[0] << ", " << 100.0*(adjacent_ptrs[1] - adjacent_ptrs[0])/edges_count << " % of all edges" << endl;
    auto degree_distribution = calculate_degree_distribution(adjacent_ptrs, vertices_count);
    cout << "degree_distribution:" << endl;
    int zero_nodes_count = 0;
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

    print_graph_memory_consumption(_graph);

    cout << "---------------------------------------------------------------------------------------------" << endl << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

pair<int, int> GraphAnalytics::calculate_power_range(int _val)
{
    double val = _val;
    int low_border = int(log2(_val));
    int high_border = int(log2(_val)) + 1;

    return (pair<int, int>(low_border, high_border));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

map<int, int> GraphAnalytics::calculate_degree_distribution(long long *_adjacent_ptrs, int _vertices_count)
{
    map<int, int> degree_distribution;

    for(int v = 0; v < _vertices_count; v++)
    {
        int connections_count = _adjacent_ptrs[v + 1] - _adjacent_ptrs[v];
        auto range = calculate_power_range(connections_count);

        if(connections_count == 0)
            degree_distribution[-1]++;
        else
            degree_distribution[range.first]++;
    }

    return degree_distribution;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void GraphAnalytics::print_graph_memory_consumption(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    double no_wights_size_in_bytes = sizeof(_TVertexValue)*vertices_count + sizeof(long long)*(vertices_count + 1) + sizeof(int)*edges_count;
    double weights_size_in_bytes = sizeof(_TEdgeWeight)*edges_count;
    double edges_in_ve = ve_vector_group_ptrs[ve_vector_segments_count - 1] - ve_vector_group_ptrs[0];
    double ve_size_in_bytes = (sizeof(int) + sizeof(_TEdgeWeight))*edges_in_ve;

    cout << "no-weights size: " << no_wights_size_in_bytes / 1e9 << " GB" << endl;
    cout << "size with weights: " << (no_wights_size_in_bytes + weights_size_in_bytes) / 1e9 << " GB" << endl;
    cout << "ve size: " << ve_size_in_bytes / 1e9 << " GB" << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
