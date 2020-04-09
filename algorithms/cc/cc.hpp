#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ConnectedComponents<_TVertexValue, _TEdgeWeight>::allocate_result_memory(int _vertices_count, int **_components)
{
    MemoryAPI::allocate_array(_components, _vertices_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ConnectedComponents<_TVertexValue, _TEdgeWeight>::free_result_memory(int *_components)
{
    MemoryAPI::free_array(_components);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ConnectedComponents<_TVertexValue, _TEdgeWeight>::component_stats(int *_components, int _vertices_count)
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

    for(auto it = size_pairs.begin(); it != size_pairs.end(); it++)
    {
        cout << "there are " << it->second << " components of size " << it->first << endl;
    }
    cout << " ---------------------------------------------------------------------------- " << endl << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ConnectedComponents<_TVertexValue, _TEdgeWeight>::performance_stats(string _name, double _time, long long _edges_count, int _iterations_count)
{
    cout << " --------------------------- " << _name << " performance stats ---------------------------- " << endl;
    cout << "wall time: " << _time*1000.0 << " ms" << endl;
    cout << "wall perf: " << _edges_count / (_time * 1e6) << " MTEPS" << endl;
    cout << "iterations count: " << _iterations_count << endl;
    cout << "perf per iteration: " << _iterations_count * (_edges_count / (_time * 1e6)) << " MTEPS" << endl;
    cout << "band per iteration: " << 5.0 * sizeof(int) * _iterations_count * (_edges_count / (_time * 1e9)) << " GB/s" << endl;
    cout << " ------------------------------------------------------------------------------------------ " << endl << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
