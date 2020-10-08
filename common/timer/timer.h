#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Timer
{
private:
    double t_start, t_end;
public:
    Timer();

    void start();

    void end();

    double get_time();

    double get_time_in_ms();

    void print_bandwidth_stats(string _name, long long _elements, double _bytes_per_element);

    void print_time_stats(string _name);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "timer.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
