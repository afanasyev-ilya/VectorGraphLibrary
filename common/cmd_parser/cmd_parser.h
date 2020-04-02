#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "parser_options.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class AlgorithmCommandOptionsParser
{
private:
    ComputeMode compute_mode;
    AlgorithmFrontierType algorithm_frontier_type;
    TraversalDirection traversal_direction;
    
    int scale;
    int avg_degree;

    int steps_count;
    
    string graph_file_name;
    
    bool check_flag;
    
    bool number_of_rounds;

    AlgorithmBFS algorithm_bfs;
    AlgorithmCC algorithm_cc;
public:
    AlgorithmCommandOptionsParser();
    
    int get_scale() { return scale; };
    int get_avg_degree() { return avg_degree; };
    ComputeMode get_compute_mode() { return compute_mode; };
    string get_graph_file_name() { return graph_file_name; };
    bool get_check_flag() { return check_flag; };
    int get_number_of_rounds() { return number_of_rounds; };
    int get_steps_count() { return steps_count; };

    AlgorithmFrontierType get_algorithm_frontier_type() {return algorithm_frontier_type;};
    TraversalDirection get_traversal_direction() {return traversal_direction;};

    AlgorithmBFS get_algorithm_bfs() {return algorithm_bfs;};
    AlgorithmCC get_algorithm_cc() {return algorithm_cc;};
    
    void parse_args(int _argc, const char * _argv[]);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "cmd_parser.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
