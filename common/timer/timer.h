#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Timer
{
private:
    double t_start, t_end;

    #ifdef __USE_GPU__
    cudaEvent_t cuda_event_start, cuda_event_stop;
    #endif
public:

    Timer();

    /* Timing API */
    inline void start();
    inline void end();

    /* Get API */
    // returns time of specified interval in seconds
    inline double get_time();

    // returns time of specified interval in milliseconds
    inline double get_time_in_ms();

    /* Print API */
    // prints effective bandwidth of specified interval
    void print_bandwidth_stats(string _name, long long _elements, double _bytes_per_element = 1);

    // prints wall time of specified interval
    void print_time_stats(string _name);

    // prints both effective bandwidth and wall time of specified interval
    void print_time_and_bandwidth_stats(string _name, long long _elements, double _bytes_per_element = 1);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "timer.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
