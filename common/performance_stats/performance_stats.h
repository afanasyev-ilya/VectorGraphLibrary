#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <map>
#include <fstream>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class PerformanceStats
{
private:
    int number_of_runs;
    double avg_time;
    double best_time;
    double worst_time;

    double inner_wall_time;
    double advance_time;
    double gather_time, scatter_time;

    double advance_ve_part_time, advance_vc_part_time, advance_collective_part_time;

    double compute_time;
    double gnf_time;
    double reduce_time;
    double reorder_time;
    double pack_time;

    void print_abstraction_stats(string _name, double _time);
    void print_detailed_advance_stats(string _name, double _time);

    double to_ms(double _time) {return 1000.0*_time;};
    double to_percent(double _time) {return 100.0*_time/inner_wall_time;};
public:
    PerformanceStats();

    void reset_timers();

    void update_advance_time(Timer &_timer);
    void update_scatter_time(Timer &_timer);
    void update_gather_time(Timer &_timer);
    void update_compute_time(Timer &_timer);
    void update_reduce_time(Timer &_timer);
    void update_gnf_time(Timer &_timer);
    void update_pack_time(Timer &_timer);
    void update_reorder_time(Timer &_timer);

    void update_advance_ve_part_time(Timer &_timer);
    void update_advance_vc_part_time(Timer &_timer);
    void update_advance_collective_part_time(Timer &_timer);

    void print_timers_stats();
    void update_timer_stats();

    void print_perf(long long _edges_count, int _k = 1);
    void print_min_perf(long long _edges_count, int _k = 1);
    void print_max_perf(long long _edges_count, int _k = 1);
    void print_avg_perf(long long _edges_count, int _k = 1);

    static void print_algorithm_performance_stats(string _name, double _time, long long _edges_count, int _iterations_count);

    static void print_algorithm_performance_stats(string _name, double _time, long long _edges_count);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

PerformanceStats performance_stats;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "performance_stats.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


