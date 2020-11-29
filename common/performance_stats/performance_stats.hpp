#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

PerformanceStats::PerformanceStats()
{
    number_of_runs = 0;
    avg_time = 0;
    best_time = 0;
    reset_timers();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::update_advance_time(Timer &_timer)
{
    #pragma omp master
    {
        advance_time += _timer.get_time();
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::update_advance_ve_part_time(Timer &_timer)
{
    #pragma omp master
    {
        advance_ve_part_time += _timer.get_time();
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::update_advance_vc_part_time(Timer &_timer)
{
    #pragma omp master
    {
        advance_vc_part_time += _timer.get_time();
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::update_advance_collective_part_time(Timer &_timer)
{
    #pragma omp master
    {
        advance_collective_part_time += _timer.get_time();
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::update_scatter_time(Timer &_timer)
{
    #pragma omp master
    {
        inner_wall_time += _timer.get_time();
        scatter_time += _timer.get_time();
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::update_gather_time(Timer &_timer)
{
    #pragma omp master
    {
        inner_wall_time += _timer.get_time();
        gather_time += _timer.get_time();
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::update_compute_time(Timer &_timer)
{
    #pragma omp master
    {
        inner_wall_time += _timer.get_time();
        compute_time += _timer.get_time();
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::update_reduce_time(Timer &_timer)
{
    #pragma omp master
    {
        inner_wall_time += _timer.get_time();
        reduce_time += _timer.get_time();
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::update_gnf_time(Timer &_timer)
{
    #pragma omp master
    {
        inner_wall_time += _timer.get_time();
        gnf_time += _timer.get_time();
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::update_pack_time(Timer &_timer)
{
    #pragma omp master
    {
        inner_wall_time += _timer.get_time();
        pack_time += _timer.get_time();
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::update_reorder_time(Timer &_timer)
{
    #pragma omp master
    {
        inner_wall_time += _timer.get_time();
        reorder_time += _timer.get_time();
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::reset_timers()
{
    inner_wall_time = 0;
    advance_time = 0;
    gather_time = 0;
    scatter_time = 0;

    advance_ve_part_time = 0;
    advance_vc_part_time = 0;
    advance_collective_part_time = 0;

    compute_time = 0;
    gnf_time = 0;
    reduce_time = 0;
    reorder_time = 0;
    pack_time = 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define SEPARATORS_LENGTH 90

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

string get_separators_upper_string(string _name)
{
    string result = " ";
    int side_size = (SEPARATORS_LENGTH - _name.length() - 2) / 2;
    for(int i = 0; i < side_size; i++)
        result += "-";
    result += " ";
    result += _name;
    result += " ";
    for(int i = 0; i < side_size; i++)
        result += "-";
    result += " ";
    return result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

string get_separators_bottom_string()
{
    string result = " ";
    for(int i = 0; i < SEPARATORS_LENGTH; i++)
        result += "-";
    result += " ";
    return result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::print_algorithm_performance_stats(string _name, double _time, long long _edges_count, int _iterations_count)
{
    cout << get_separators_upper_string(_name) << endl;
    cout << "wall time: " << _time*1000.0 << " ms" << endl;
    cout << "wall perf: " << _edges_count / (_time * 1e6) << " MTEPS" << endl;
    cout << "iterations count: " << _iterations_count << endl;
    cout << "perf per iteration: " << _iterations_count * (_edges_count / (_time * 1e6)) << " MTEPS" << endl;
    cout << "band per iteration: " << INT_ELEMENTS_PER_EDGE * sizeof(int) * _iterations_count * (_edges_count / (_time * 1e9)) << " GB/s" << endl;
    cout << get_separators_bottom_string() << endl << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void PerformanceStats::print_algorithm_performance_stats(string _name, double _time, long long _edges_count)
{
    cout << get_separators_upper_string(_name) << endl;
    cout << "wall time: " << _time*1000.0 << " ms" << endl;
    cout << "wall perf: " << _edges_count / (_time * 1e6) << " MTEPS" << endl;
    cout << get_separators_bottom_string() << endl << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::print_timers_stats()
{
    cout << endl;
    print_abstraction_stats("Inner wall    ", inner_wall_time);
    print_abstraction_stats("Advance       ", advance_time);
    #ifdef __USE_NEC_SX_AURORA__
    print_detailed_advance_stats("Advance (ve) time        ", advance_ve_part_time);
    print_detailed_advance_stats("Advance (vc) time        ", advance_vc_part_time);
    print_detailed_advance_stats("Advance (collective) time", advance_collective_part_time);
    #endif
    print_abstraction_stats("Gather        ", gather_time);
    print_abstraction_stats("Scatter       ", scatter_time);
    print_abstraction_stats("Compute       ", compute_time);
    print_abstraction_stats("Reduce        ", reduce_time);
    print_abstraction_stats("GNF           ", gnf_time);
    print_abstraction_stats("Reorder       ", reorder_time);
    print_abstraction_stats("Pack          ", pack_time);
    cout << endl;

    update_timer_stats();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::update_timer_stats()
{
    if(best_time < inner_wall_time)
        best_time = inner_wall_time;
    avg_time += inner_wall_time;
    number_of_runs++;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::print_max_perf(long long _edges_count, int _k)
{
    cout << "MAX_PERF: " << _k*(_edges_count / (best_time * 1e6)) << " MTEPS (among " << number_of_runs << " runs)" << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::print_avg_perf(long long _edges_count, int _k)
{
    avg_time /= number_of_runs;
    cout << "AVG_PERF: " << _k*(_edges_count / (avg_time * 1e6)) << " MTEPS (among " << number_of_runs << " runs)" << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::print_abstraction_stats(string _name, double _time)
{
    if(_time > 0)
        cout << _name << " : " << to_ms(_time) << " (ms), " << to_percent(_time) << "%" << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::print_detailed_advance_stats(string _name, double _time)
{
    if(_time > 0)
        cout << "    "  << _name << " : " << to_ms(_time) << " (ms), " << to_percent(_time) << "%" << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


