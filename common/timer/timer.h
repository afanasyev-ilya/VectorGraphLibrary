#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Timer
{
private:
    double t_start, t_end;
public:

    Timer();

    /* Timing API */
    void start();
    void end();

    /* Get API */
    // returns time of specified interval in seconds
    double get_time();

    // returns time of specified interval in milliseconds
    double get_time_in_ms();

    /* Print API */
    // prints effective bandwidth of specified interval
    void print_bandwidth_stats(string _name, long long _elements, double _bytes_per_element);

    // prints wall time of specified interval
    void print_time_stats(string _name);

    // prints both effective bandwidth and wall time of specified interval
    void print_time_and_bandwidth_stats(string _name, long long _elements, double _bytes_per_element);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef __CUDA_INCLUDE__
#include "timer.hpp"
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
