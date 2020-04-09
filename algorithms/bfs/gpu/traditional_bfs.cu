#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <vector>
#include "../../../common/gpu_API/cuda_error_handling.h"
#include "../../../architectures.h"
#include <cfloat>
#include "../../../graph_representations/base_graph.h"
#include "../../../graph_representations/edges_list_graph/edges_list_graph.h"
#include "../../../graph_representations/extended_CSR_graph/extended_CSR_graph.h"
#include "../change_state/change_state.h"

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void gpu_direction_optimising_bfs_wrapper<int, float>(ExtendedCSRGraph<int, float> &_graph,
                                                      int *_levels,
                                                      int _source_vertex,
                                                      int &_iterations_count)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);
    GraphStructure graph_structure = check_graph_structure(_graph);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template void gpu_direction_optimising_bfs_wrapper<int, float>(ExtendedCSRGraph<int, float> &_graph, int *_levels,
                                                               int _source_vertex, int &_iterations_count);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

