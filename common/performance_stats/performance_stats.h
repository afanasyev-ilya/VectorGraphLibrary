#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <map>
#include <fstream>
#include <limits>

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
    double non_api_time;

    #ifdef __USE_MPI__
    double MPI_time;
    double MPI_functions_time;
    int get_mpi_rank();
    #endif

    size_t bytes_requested;
    size_t edges_visited;

    void print_abstraction_stats(string _name, double _time);
    void print_detailed_advance_stats(string _name, double _time);

    double to_ms(double _time) {return 1000.0*_time;};
    double to_percent(double _time) {return 100.0*_time/inner_wall_time;};
public:
    PerformanceStats();

    void reset_timers();

    inline void update_bytes_requested(size_t bytes);
    inline void update_edges_visited(size_t edges);
    inline void update_graph_processing_stats(size_t _bytes, size_t _edges);

    inline void update_advance_time(Timer &_timer);
    inline void update_scatter_time(Timer &_timer);
    inline void update_gather_time(Timer &_timer);
    inline void update_compute_time(Timer &_timer);
    inline void update_reduce_time(Timer &_timer);
    inline void update_gnf_time(Timer &_timer);
    inline void update_pack_time(Timer &_timer);
    inline void update_reorder_time(Timer &_timer);
    inline void update_non_api_time(Timer &_timer);

    #ifdef __USE_MPI__
    inline void update_MPI_time(Timer &_timer);
    inline void update_MPI_functions_time(Timer &_timer);
    #endif

    inline void update_advance_ve_part_time(Timer &_timer);
    inline void update_advance_vc_part_time(Timer &_timer);
    inline void update_advance_collective_part_time(Timer &_timer);

    inline void fast_update_advance_stats(double _wall_time,
                                          double _ve_part_time,
                                          double _vc_part_time,
                                          double _collective_part_time,
                                          size_t _bytes,
                                          size_t _edges);

    void print_timers_stats();
    void update_timer_stats();

    void print_perf(long long _edges_count, int _k = 1);

    void reset_perf_stats();

    double get_max_perf(long long _edges_count, int _k = 1);
    double get_min_perf(long long _edges_count, int _k = 1);
    double get_avg_perf(long long _edges_count, int _k = 1);
    double get_sustained_bandwidth() {return bytes_requested / (inner_wall_time * 1e9);};
    double get_edges_rate() {return edges_visited / (inner_wall_time * 1e6); }
    double get_inner_time() { return inner_wall_time; };

    void print_min_perf(long long _edges_count, int _k = 1);
    void print_max_perf(long long _edges_count, int _k = 1);
    void print_avg_perf(long long _edges_count, int _k = 1);

    void print_algorithm_performance_stats(string _name, double _time, long long _edges_count);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

PerformanceStats performance_stats;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "performance_stats.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


