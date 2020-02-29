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

enum AlgorithmFrontierType {
    ALL_ACTIVE = 1,
    PARTIAL_ACTIVE = 0
};

enum TraversalDirection {
    PUSH_TRAVERSAL = 1,
    PULL_TRAVERSAL = 0
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum VerticesState
{
    VERTICES_SORTED = 1,
    VERTICES_UNSORTED = 0,
    VERTICES_RANDOM_SHUFFLED = 2
};

enum EdgesState
{
    EDGES_SORTED = 1,
    EDGES_UNSORTED = 0,
    EDGES_RANDOM_SHUFFLED = 2
};

enum MultipleArcsState
{
    MULTIPLE_ARCS_PRESENT = 1,
    MULTIPLE_ARCS_REMOVED = 0
};

enum AlgorithmBFS
{
    TOP_DOWN_BFS_ALGORITHM = 0,
    BOTTOM_UP_BFS_ALGORITHM = 1,
    DIRECTION_OPTIMISING_BFS_ALGORITHM = 2
};

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
    
    void parse_args(int _argc, const char * _argv[]);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "cmd_parser.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* cmd_parser_h */
