#pragma once

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

enum AlgorithmTraversalType {
    PUSH_TRAVERSAL = 1,
    PULL_TRAVERSAL = 0,
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
    DIRECTION_OPTIMIZING_BFS_ALGORITHM = 2
};

enum AlgorithmCC
{
    SHILOACH_VISHKIN_ALGORITHM = 0,
    BFS_BASED_ALGORITHM = 1
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
