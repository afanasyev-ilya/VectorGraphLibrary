#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#include <fstream>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class PerformanceData
{
private:
    string file_name;
    string short_file_name;
public:
    PerformanceData(string _operation_name)
    {
        file_name = _operation_name + "_performance_data.txt";
        short_file_name = _operation_name + "_performance_data_short.txt";
    }

    void save_performance_data(string _graph_name, double _perf)
    {
        ofstream perf_file;
        perf_file.open(file_name.c_str(), std::ios_base::app);
        perf_file << _graph_name << ": " << _perf << " MTEPS" << endl;
        perf_file.close();

        ofstream short_perf_file;
        short_perf_file.open(short_file_name.c_str(), std::ios_base::app);
        short_perf_file << _perf << endl;
        short_perf_file.close();
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "performance_data.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


