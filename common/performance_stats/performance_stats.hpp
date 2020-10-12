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

void PerformanceStats::reset_API_performance_timers()
{
    INNER_WALL_TIME = 0;
    INNER_ADVANCE_TIME = 0;
    INNER_COMPUTE_TIME = 0;
    INNER_GNF_TIME = 0;
    INNER_FILTER_TIME = 0;
    INNER_REDUCE_TIME = 0;
    INNER_PACK_TIME = 0;
    INNER_WALL_WORK = 0;

    #ifdef __USE_NEC_SX_AURORA__
    DETAILED_ADVANCE_PART_1_NEC_TIME = 0;
    DETAILED_ADVANCE_PART_2_NEC_TIME = 0;
    DETAILED_ADVANCE_PART_3_NEC_TIME = 0;
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::print_API_performance_timers(long long _edges_count)
{
    cout << "test: " << INT_ELEMENTS_PER_EDGE << endl;
    cout << "INNER_WALL_TIME: " << INNER_WALL_TIME * 1000 << " ms" << endl;
    cout << "INNER perf: " << _edges_count / (INNER_WALL_TIME * 1e6) << " MTEPS" << endl;
    cout << "INNER_ADVANCE_TIME: " << int(100.0 * INNER_ADVANCE_TIME / INNER_WALL_TIME) << " %" << endl;

    double advance_bw = sizeof(int)*INT_ELEMENTS_PER_EDGE*INNER_WALL_WORK / (1e9*INNER_ADVANCE_TIME);
    cout << "ADVANCE_BANDWIDTH: " << advance_bw << " GB/s" << endl;
    #ifdef __USE_NEC_SX_AURORA__
    cout << "         DETAILED_ADVANCE_PART_1_NEC_TIME: " << int(100.0 * DETAILED_ADVANCE_PART_1_NEC_TIME / INNER_WALL_TIME) << " % (" << 1000.0*DETAILED_ADVANCE_PART_1_NEC_TIME << " ms)" << endl;
    cout << "         DETAILED_ADVANCE_PART_2_NEC_TIME: " << int(100.0 * DETAILED_ADVANCE_PART_2_NEC_TIME / INNER_WALL_TIME) << " % (" << 1000.0*DETAILED_ADVANCE_PART_2_NEC_TIME << " ms)" << endl;
    cout << "         DETAILED_ADVANCE_PART_3_NEC_TIME: " << int(100.0 * DETAILED_ADVANCE_PART_3_NEC_TIME / INNER_WALL_TIME) << " % (" << 1000.0*DETAILED_ADVANCE_PART_3_NEC_TIME << " ms)" << endl;
    #endif

    cout << "INNER_COMPUTE_TIME: " << int(100.0 * INNER_COMPUTE_TIME / INNER_WALL_TIME) << " %" << endl;
    cout << "INNER_GNF_TIME: " << int(100.0 * INNER_GNF_TIME / INNER_WALL_TIME) << " % (" << 1000.0*INNER_GNF_TIME << " ms)" << endl;
    cout << "INNER_REDUCE_TIME: " << int(100.0 * INNER_REDUCE_TIME / INNER_WALL_TIME) << " %" << endl;
    cout << "INNER_PACK_TIME: " << int(100.0 * INNER_PACK_TIME / INNER_WALL_TIME) << " %" << endl;
    cout << "INNER_FILTER_TIME: " << int(100.0 * INNER_FILTER_TIME / INNER_WALL_TIME) << " %" << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


