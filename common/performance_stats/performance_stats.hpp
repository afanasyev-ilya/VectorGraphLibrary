#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

PerformanceStats::PerformanceStats()
{
    reset_perf_stats();
    reset_timers();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::fast_update_advance_stats(double _wall_time,
                                                 double _ve_part_time,
                                                 double _vc_part_time,
                                                 double _collective_part_time,
                                                 size_t _bytes,
                                                 size_t _edges)
{
    advance_time += _wall_time;
    advance_ve_part_time += _ve_part_time;
    advance_vc_part_time += _vc_part_time;
    advance_collective_part_time += _collective_part_time;
    inner_wall_time += _wall_time;

    bytes_requested += _bytes;
    edges_visited += _edges;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::update_advance_time(Timer &_timer)
{
    #pragma omp single
    {
        advance_time += _timer.get_time();
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::update_advance_ve_part_time(Timer &_timer)
{
    #pragma omp single
    {
        inner_wall_time += _timer.get_time();
        advance_ve_part_time += _timer.get_time();
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::update_advance_vc_part_time(Timer &_timer)
{
    #pragma omp single
    {
        advance_vc_part_time += _timer.get_time();
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::update_advance_collective_part_time(Timer &_timer)
{
    #pragma omp single
    {
        advance_collective_part_time += _timer.get_time();
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::update_scatter_time(Timer &_timer)
{
    #pragma omp single
    {
        scatter_time += _timer.get_time();
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::update_gather_time(Timer &_timer)
{
    #pragma omp single
    {
        gather_time += _timer.get_time();
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::update_compute_time(Timer &_timer)
{
    #pragma omp single
    {
        inner_wall_time += _timer.get_time();
        compute_time += _timer.get_time();
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::update_reduce_time(Timer &_timer)
{
    #pragma omp single
    {
        inner_wall_time += _timer.get_time();
        reduce_time += _timer.get_time();
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::update_gnf_time(Timer &_timer)
{
    #pragma omp single
    {
        inner_wall_time += _timer.get_time();
        gnf_time += _timer.get_time();
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::update_pack_time(Timer &_timer)
{
    #pragma omp single
    {
        inner_wall_time += _timer.get_time();
        pack_time += _timer.get_time();
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::update_reorder_time(Timer &_timer)
{
    #pragma omp single
    {
        inner_wall_time += _timer.get_time();
        reorder_time += _timer.get_time();
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::update_non_api_time(Timer &_timer)
{
    #pragma omp single
    {
        inner_wall_time += _timer.get_time();
        non_api_time += _timer.get_time();
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_MPI__
void PerformanceStats::update_MPI_time(Timer &_timer)
{
    #pragma omp single
    {
        inner_wall_time += _timer.get_time();
        MPI_time += _timer.get_time();
    }
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_MPI__
void PerformanceStats::update_MPI_functions_time(Timer &_timer)
{
    #pragma omp single
    {
        MPI_functions_time += _timer.get_time();
    }
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::update_bytes_requested(size_t _bytes)
{
    #pragma omp single
    {
        bytes_requested += _bytes;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::update_edges_visited(size_t _edges)
{
    #pragma omp single
    {
        edges_visited += _edges;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::update_graph_processing_stats(size_t _bytes, size_t _edges)
{
    #pragma omp single
    {
        bytes_requested += _bytes;
        edges_visited += _edges;
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

    non_api_time = 0;

    #ifdef __USE_MPI__
    MPI_time = 0;
    MPI_functions_time = 0;
    #endif

    bytes_requested = 0;
    edges_visited = 0;
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

void PerformanceStats::print_algorithm_performance_stats(string _name, double _time, long long _edges_count)
{
    bool print_condition = true;
    #ifdef __USE_MPI__
    MPI_Barrier(MPI_COMM_WORLD);
    if(get_mpi_rank() != 0)
        print_condition = false;
    #endif

    if(print_condition)
    {
        cout << get_separators_upper_string(_name) << endl;
        cout << "Wall time: " << _time*1000.0 << " ms" << endl;
        cout << "Wall (graph500) perf: " << _edges_count / (_time * 1e6) << " MTEPS" << endl;
        cout << get_separators_bottom_string() << endl << endl;
    }

    #ifdef __USE_MPI__
    MPI_Barrier(MPI_COMM_WORLD);
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::print_timers_stats()
{
    bool print_condition = true;
    #ifdef __USE_MPI__
    MPI_Barrier(MPI_COMM_WORLD);
    if(get_mpi_rank() != 0)
        print_condition = false;
    #endif

    if(print_condition)
    {
        cout << endl;
        print_abstraction_stats("Inner wall    ", inner_wall_time);
        print_abstraction_stats("Advance       ", advance_time);
        #if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
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
        print_abstraction_stats("Non-API       ", non_api_time);
        #ifdef __USE_MPI__
        print_abstraction_stats("MPI           ", MPI_time);
        print_abstraction_stats("MPI functions ", MPI_functions_time);
        print_abstraction_stats("MPI preprocess time ", MPI_time - MPI_functions_time);
        #endif
        cout << endl;

        cout << "total bandwidth: " << get_sustained_bandwidth() << " GB/s" << endl;
        cout << "edges rate: " << get_edges_rate() << " MTEPS" << endl;
        cout << "edges visited: " << edges_visited << endl;
        cout << endl;
    }

    #ifdef __USE_MPI__
    MPI_Barrier(MPI_COMM_WORLD);
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::update_timer_stats()
{
    if(best_time > inner_wall_time)
        best_time = inner_wall_time;

    if(worst_time < inner_wall_time)
        worst_time = inner_wall_time;

    avg_time += inner_wall_time;
    number_of_runs++;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::print_perf(long long _edges_count, int _k)
{
    print_min_perf(_edges_count, _k);
    print_avg_perf(_edges_count, _k);
    print_max_perf(_edges_count, _k);

    #ifdef __USE_MPI__
    if(get_mpi_rank() == 0)
    {
        ofstream outfile;
        outfile.open("MPI_scale_perf.txt", std::ios_base::app);
        outfile << get_max_perf(_edges_count, _k) << endl;
    }
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::reset_perf_stats()
{
    number_of_runs = 0;
    avg_time = 0;
    best_time = std::numeric_limits<double>::max();
    worst_time = 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double PerformanceStats::get_max_perf(long long _edges_count, int _k)
{
    return _k*(_edges_count / (best_time * 1e6));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::print_max_perf(long long _edges_count, int _k)
{
    bool print_condition = true;
    #ifdef __USE_MPI__
    MPI_Barrier(MPI_COMM_WORLD);
    if(get_mpi_rank() != 0)
        print_condition = false;
    #endif

    if(print_condition)
    {
        cout << "MAX_PERF: " << get_max_perf(_edges_count, _k) << " MTEPS (among " << number_of_runs << " runs)"
             << endl;
    }

    #ifdef __USE_MPI__
    MPI_Barrier(MPI_COMM_WORLD);
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double PerformanceStats::get_min_perf(long long _edges_count, int _k)
{
    return _k*(_edges_count / (worst_time * 1e6));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::print_min_perf(long long _edges_count, int _k)
{
    bool print_condition = true;
    #ifdef __USE_MPI__
    MPI_Barrier(MPI_COMM_WORLD);
    if(get_mpi_rank() != 0)
        print_condition = false;
    #endif

    if(print_condition)
    {
        cout << "MIN_PERF: " << get_min_perf(_edges_count, _k) << " MTEPS (among " << number_of_runs << " runs)" << endl;
    }

    #ifdef __USE_MPI__
    MPI_Barrier(MPI_COMM_WORLD);
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double PerformanceStats::get_avg_perf(long long _edges_count, int _k)
{
    double local_avg_time = avg_time / number_of_runs;
    return _k*(_edges_count / (local_avg_time * 1e6));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::print_avg_perf(long long _edges_count, int _k)
{
    bool print_condition = true;
    #ifdef __USE_MPI__
    MPI_Barrier(MPI_COMM_WORLD);
    if(get_mpi_rank() != 0)
        print_condition = false;
    #endif

    if(print_condition)
    {
        cout << "AVG_PERF: " << get_avg_perf(_edges_count, _k) << " MTEPS (among " << number_of_runs << " runs)" << endl;
    }

    #ifdef __USE_MPI__
    MPI_Barrier(MPI_COMM_WORLD);
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::print_abstraction_stats(string _name, double _time)
{
    #ifdef __USE_MPI__
    if(_time > 0)
        cout << _name << " : " << to_ms(_time) << " (ms), " << to_percent(_time) << "% (proc №" <<
        get_mpi_rank() << ")" << endl;
    #else
    if(_time > 0)
        cout << _name << " : " << to_ms(_time) << " (ms), " << to_percent(_time) << "%" << endl;
    #endif

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PerformanceStats::print_detailed_advance_stats(string _name, double _time)
{
    #ifdef __USE_MPI__
    if(_time > 0)
        cout << "    "  << _name << " : " << to_ms(_time) << " (ms), " << to_percent(_time) << "% (proc №" <<
        get_mpi_rank() << ")" << endl;
    #else
    if(_time > 0)
        cout << "    "  << _name << " : " << to_ms(_time) << " (ms), " << to_percent(_time) << "%" << endl;
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_MPI__
int PerformanceStats::get_mpi_rank()
{
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    return mpi_rank;
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



