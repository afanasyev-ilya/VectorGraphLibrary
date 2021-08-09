#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <string>

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define VGL_PACK_TYPE long long

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class VectorCSRGraph;
class EdgesListGraph;
class VGL_Graph;
class ShardedCSRGraph;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Frontier;
class FrontierNEC;
class VGL_Frontier;
class FrontierMulticore;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
class VerticesArray;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
class EdgesArray;
template <typename _T>
class EdgesArray_EL;
template <typename _T>
class EdgesArray_Sharded;
template <typename _T>
class EdgesArray_Vect;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class GraphAnalytics;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum GraphType
{
    VGL_GRAPH = 0,
    VECTOR_CSR_GRAPH = 1,
    EDGES_LIST_GRAPH = 2,
    CSR_GRAPH = 3
};

string get_graph_type_name(GraphType _type)
{
    if(_type == VGL_GRAPH)
        return "VGL_GRAPH";
    else if(_type == VECTOR_CSR_GRAPH)
        return "VECTOR_CSR_GRAPH";
    else if(_type == EDGES_LIST_GRAPH)
        return "EDGES_LIST_GRAPH";
    else if(_type == CSR_GRAPH)
        return "CSR_GRAPH";
    else
        return "UNKNOWN";
}

string get_graph_extension(GraphType _type)
{
    if(_type == VGL_GRAPH)
        return ".vgl";
    else if(_type == VECTOR_CSR_GRAPH)
        return ".vcsr";
    else if(_type == EDGES_LIST_GRAPH)
        return ".el";
    else if(_type == CSR_GRAPH)
        return ".csr";
    else
        return ".unknown";
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum ObjectType
{
    GRAPH = 0,
    FRONTIER = 1,
    VERTICES_ARRAY = 2,
    EDGES_ARRAY = 3
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum TraversalDirection {
    SCATTER = 0, // outgoing is always first
    GATHER = 1, // incoming is second
    ORIGINAL = 2, // for datastructures
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum SupportedDirection {
    USE_BOTH = 0,
    USE_SCATTER_ONLY = 1,
    USE_GATHER_ONLY = 2
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum EdgesStorageType {
    CSR_STORAGE = 0, // MUST BE 0 for correct CSR format
    VE_STORAGE = 1
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum REDUCE_TYPE
{
    REDUCE_SUM = 0,
    REDUCE_MAX = 1,
    REDUCE_MIN = 1,
    REDUCE_AVG = 3
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum FrontierClassType {
    BASE_FRONTIER = 0,
    VECTOR_CSR_FRONTIER = 1,
    GENERAL_FRONTIER = 2
};

enum FrontierSparsityType {
    ALL_ACTIVE_FRONTIER = 2,
    SPARSE_FRONTIER = 1,
    DENSE_FRONTIER = 0
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define IN_FRONTIER_FLAG 1
#define NOT_IN_FRONTIER_FLAG 0

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum VisualisationMode
{
    VISUALISE_AS_DIRECTED = 0,
    VISUALISE_AS_UNDIRECTED = 1
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

