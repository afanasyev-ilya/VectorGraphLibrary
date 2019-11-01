//
//  cmd_parser.h
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 07/09/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef cmd_parser_h
#define cmd_parser_h

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum ComputeMode {
    GENERATE_NEW_GRAPH,
    LOAD_GRAPH_FROM_FILE
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class AlgorithmCommandOptionsParser
{
private:
    ComputeMode compute_mode;
    
    int scale;
    int avg_degree;
    
    string graph_file_name;
    
    bool check_flag;
    
    bool number_of_rounds;
public:
    AlgorithmCommandOptionsParser();
    
    int get_scale() { return scale; };
    int get_avg_degree() { return avg_degree; };
    ComputeMode get_compute_mode() { return compute_mode; };
    string get_graph_file_name() { return graph_file_name; };
    bool get_check_flag() { return check_flag; };
    int get_number_of_rounds() { return number_of_rounds; };
    
    void parse_args(int _argc, const char * _argv[]);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "cmd_parser.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* cmd_parser_h */
