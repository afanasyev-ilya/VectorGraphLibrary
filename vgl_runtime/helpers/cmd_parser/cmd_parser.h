#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "parser_options.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Parser
{
private:
    ComputeMode compute_mode;
    AlgorithmFrontierType algorithm_frontier_type;
    AlgorithmTraversalType traversal_direction;
    GraphStorageFormat graph_storage_format;
    GraphStorageOptimizations graph_storage_optimizations;

    PartitioningAlgorithm partitioning_mode;
    
    int scale;
    int avg_degree;
    SyntheticGraphType synthetic_graph_type;

    string graph_file_name;

    bool check_flag;

    int number_of_rounds;

    AlgorithmBFS algorithm_bfs;
    AlgorithmCC algorithm_cc;
    AlgorithmTC algorithm_tc;

    int store_walk_paths;
    int walk_vertices_percent;

    int device_num;

    bool convert;
    string convert_name;
public:
    Parser();
    
    int get_scale() { return scale; };
    int get_avg_degree() { return avg_degree; };
    ComputeMode get_compute_mode() { return compute_mode; };
    string get_graph_file_name() { return graph_file_name; };
    bool get_check_flag() { return check_flag; };
    int get_number_of_rounds() { return number_of_rounds; };
    int get_device_num() { return device_num; };
    GraphStorageFormat get_graph_storage_format() {return graph_storage_format;};
    GraphStorageOptimizations get_graph_storage_optimizations() {return graph_storage_optimizations;};
    PartitioningAlgorithm get_partitioning_mode() {return partitioning_mode;};

    bool get_convert() {return convert;};
    string get_convert_name() {return convert_name;};

    int get_store_walk_paths() {return store_walk_paths;};
    int get_walk_vertices_percent() {return walk_vertices_percent;};

    SyntheticGraphType get_synthetic_graph_type() {return synthetic_graph_type;};

    AlgorithmFrontierType get_algorithm_frontier_type() {return algorithm_frontier_type;};
    AlgorithmTraversalType get_traversal_direction() {return traversal_direction;};

    AlgorithmBFS get_algorithm_bfs() {return algorithm_bfs;};
    AlgorithmCC get_algorithm_cc() {return algorithm_cc;};
    AlgorithmTC get_algorithm_tc() {return algorithm_tc;};
    
    void parse_args(int _argc, char **_argv);

    static TraversalDirection convert_traversal_type(AlgorithmTraversalType _algo_type);
    static AlgorithmTraversalType convert_traversal_type(TraversalDirection _direction_type);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "cmd_parser.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
