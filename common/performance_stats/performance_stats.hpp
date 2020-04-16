#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::save_performance_to_file(string _operation_name, string _graph_name, double _perf)
{
    string file_name = _operation_name + "_performance_data.txt";
    string short_file_name = _operation_name + "_performance_data_short.txt";

    ofstream perf_file;
    perf_file.open(file_name.c_str(), std::ios_base::app);
    perf_file << _graph_name << ": " << _perf << " MTEPS" << endl;
    perf_file.close();

    ofstream short_perf_file;
    short_perf_file.open(short_file_name.c_str(), std::ios_base::app);
    short_perf_file << _perf << endl;
    short_perf_file.close();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::component_stats(int *_components, int _vertices_count)
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

void PerformanceStats::print_performance_stats(string _name, double _time, long long _edges_count, int _iterations_count)
{
    cout << " --------------------------- " << _name << " performance stats --------------------------- " << endl;
    cout << "wall time: " << _time*1000.0 << " ms" << endl;
    cout << "wall perf: " << _edges_count / (_time * 1e6) << " MTEPS" << endl;
    cout << "iterations count: " << _iterations_count << endl;
    cout << "perf per iteration: " << _iterations_count * (_edges_count / (_time * 1e6)) << " MTEPS" << endl;
    cout << "band per iteration: " << INT_ELEMENTS_PER_EDGE * sizeof(int) * _iterations_count * (_edges_count / (_time * 1e9)) << " GB/s" << endl;
    cout << " ----------------------------------------------------------------------------------------- " << endl << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
