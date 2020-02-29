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
void ConnectedComponents<_TVertexValue, _TEdgeWeight>::print_component_stats(int *_components, int _vertices_count)
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

    sort(size_pairs.begin(), size_pairs.end(), [=](std::pair<int, int>& a, std::pair<int, int>& b)
    { return a.first >= b.first; } );

    for(auto it = size_pairs.begin(); it != size_pairs.end(); it++)
    {
        cout << "there are " << it->second << " components of size " << it->first << endl;
    }
    cout << " ---------------------------------------------------------------------------- " << endl << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
