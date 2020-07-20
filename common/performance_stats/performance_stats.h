#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <map>
#include <fstream>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// performance debug variables
double INNER_WALL_TIME = 0;
double INNER_ADVANCE_TIME = 0;
double DETAILED_ADVANCE_PART_1_NEC_TIME = 0;
double DETAILED_ADVANCE_PART_2_NEC_TIME = 0;
double DETAILED_ADVANCE_PART_3_NEC_TIME = 0;
double INNER_COMPUTE_TIME = 0;
double INNER_GNF_TIME = 0;
double INNER_FILTER_TIME = 0;
double INNER_REDUCE_TIME = 0;
double INNER_PACK_TIME = 0;
double INNER_WALL_WORK = 0;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class PerformanceStats
{
public:
    static void reset_API_performance_timers();

    static void print_API_performance_timers(long long _edges_count);

    static void save_performance_to_file(string _operation_name, string _graph_name, double _perf);

    static void print_performance_stats(string _name, double _time, long long _edges_count, int _iterations_count);

    static void component_stats(int *_components, int _vertices_count);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "performance_stats.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


