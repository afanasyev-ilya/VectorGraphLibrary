#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../../../graph_processing_API/gpu/cuda_API_include.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void gpu_lp_wrapper(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int *_labels)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);
    GraphPrimitivesGPU graph_API;
    FrontierGPU frontier(_graph.get_vertices_count());

    int *gathered_labels;
    MemoryAPI::allocate_device_array(&gathered_labels, edges_count);

    frontier.set_all_active();

    auto init_op = [_labels] __device__ (int src_id, int connections_count) {
        _labels[src_id] = src_id;
    };

    graph_API.compute(_graph, frontier, init_op);

    auto gather_edge_op = [_labels, gathered_labels] __device__(int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos){
        int dst_label = __ldg(&_labels[dst_id]);
        gathered_labels[global_edge_pos] = dst_label;
    };
    graph_API.advance(_graph, frontier, gather_edge_op);

    MemoryAPI::free_device_array(gathered_labels);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template void gpu_lp_wrapper<int, float>(ExtendedCSRGraph<int, float> &_graph, int *_labels);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
