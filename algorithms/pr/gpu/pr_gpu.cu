#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 5.0

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../../../graph_processing_API/gpu/cuda_API_include.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void page_rank_wrapper(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                       float *_page_ranks,
                       float _convergence_factor,
                       int _max_iterations,
                       int &_iterations_count,
                       TraversalDirection _traversal_direction)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);
    GraphPrimitivesGPU graph_API;
    FrontierGPU frontier(vertices_count);

    frontier.set_all_active();

    int   *number_of_loops;
    int   *incoming_degrees_without_loops;
    float *old_page_ranks;
    float *reversed_degrees;
    MemoryAPI::allocate_device_array(&number_of_loops, vertices_count);
    MemoryAPI::allocate_device_array(&incoming_degrees_without_loops, vertices_count);
    MemoryAPI::allocate_device_array(&old_page_ranks, vertices_count);
    MemoryAPI::allocate_device_array(&reversed_degrees, vertices_count);

    float d = 0.85;
    float k = (1.0 - d) / ((float)vertices_count);

    frontier.set_all_active();

    auto init_data = [_page_ranks, number_of_loops, vertices_count] __device__(int src_id, int position_in_frontier, int connections_count)
    {
        _page_ranks[src_id] = 1.0/vertices_count;
        number_of_loops[src_id] = 0;
    };
    graph_API.compute(_graph, frontier, init_data);

    auto calculate_number_of_loops = [number_of_loops]__device__(int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos, int position_in_frontier)
    {
        if(src_id == dst_id)
        {
            number_of_loops[src_id] += 1;
        }
    };

    graph_API.advance(_graph, frontier, calculate_number_of_loops);

    if(_traversal_direction == PUSH_TRAVERSAL)
    {
        auto calculate_degrees_without_loops_push = [incoming_degrees_without_loops, incoming_degrees, number_of_loops] __device__(int src_id, int position_in_frontier, int connections_count)
        {
            incoming_degrees_without_loops[src_id] = connections_count - number_of_loops[src_id];
        };
        graph_API.compute(_graph, frontier, calculate_degrees_without_loops_push);
    }
    else if(_traversal_direction == PULL_TRAVERSAL)
    {
        auto calculate_degrees_without_loops_pull = [incoming_degrees_without_loops, incoming_degrees, number_of_loops] __device__(int src_id, int position_in_frontier, int connections_count)
        {
            incoming_degrees_without_loops[src_id] = incoming_degrees[src_id] - number_of_loops[src_id];
        };
        graph_API.compute(_graph, frontier, calculate_degrees_without_loops_pull);
    }

    auto calculate_reversed_degrees = [reversed_degrees, incoming_degrees_without_loops] __device__(int src_id, int position_in_frontier, int connections_count)
    {
        reversed_degrees[src_id] = 1.0 / incoming_degrees_without_loops[src_id];
    };
    graph_API.compute(_graph, frontier, calculate_reversed_degrees);

    int current_iteration = 0;
    for(int iterations_count = 0; iterations_count < _max_iterations; iterations_count++)
    {
        auto save_old_ranks = [old_page_ranks, _page_ranks] __device__(int src_id, int position_in_frontier, int connections_count)
        {
            old_page_ranks[src_id] = _page_ranks[src_id];
            _page_ranks[src_id] = 0;
        };
        graph_API.compute(_graph, frontier, save_old_ranks);

        auto reduce_dangling_input = [incoming_degrees_without_loops, old_page_ranks, vertices_count] __device__(int src_id, int position_in_frontier, int connections_count)->double
        {
            float result = 0.0;
            if(incoming_degrees_without_loops[src_id] == 0)
            {
                result = old_page_ranks[src_id] / vertices_count;
            }
            return result;
        };
        double dangling_input = graph_API.reduce<double>(_graph, frontier, reduce_dangling_input, REDUCE_SUM);

        auto edge_op_pull = [_page_ranks, old_page_ranks, reversed_degrees]__device__(int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos, int position_in_frontier)
        {
            float dst_rank = old_page_ranks[dst_id];
            float reversed_dst_links_num = reversed_degrees[dst_id];

            if(src_id != dst_id)
                atomicAdd(&_page_ranks[src_id], dst_rank * reversed_dst_links_num);
        };

        if(_traversal_direction == PUSH_TRAVERSAL)
        {
            auto edge_op_push = [_page_ranks, old_page_ranks, reversed_degrees]__device__(int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos, int position_in_frontier)
            {
                float src_rank = old_page_ranks[src_id];
                float reversed_src_links_num = reversed_degrees[src_id];

                if(src_id != dst_id)
                    atomicAdd(&_page_ranks[dst_id], src_rank * reversed_src_links_num);
            };

            graph_API.advance(_graph, frontier, edge_op_push);
        }
        else if(_traversal_direction == PULL_TRAVERSAL)
        {
            auto edge_op_pull = [_page_ranks, old_page_ranks, reversed_degrees]__device__(int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos, int position_in_frontier)
            {
                float dst_rank = old_page_ranks[dst_id];
                float reversed_dst_links_num = reversed_degrees[dst_id];

                if(src_id != dst_id)
                    atomicAdd(&_page_ranks[src_id], dst_rank * reversed_dst_links_num);
            };

            graph_API.advance(_graph, frontier, edge_op_pull);
        }

        auto reduce_ranks_sum = [_page_ranks]__device__(int src_id, int position_in_frontier, int connections_count)->double
        {
            return _page_ranks[src_id];
        };
        double ranks_sum = graph_API.reduce<double>(_graph, frontier, reduce_ranks_sum, REDUCE_SUM);
        current_iteration++;

        /*auto reduce_error = [_page_ranks, old_page_ranks]__device__(int src_id, int position_in_frontier, int connections_count)->double
        {
            return fabs(_page_ranks[src_id] - old_page_ranks[src_id]);
        };

        double error = graph_API.reduce<double>(_graph, frontier, reduce_error, REDUCE_SUM);
        if(error < vertices_count*_convergence_factor)
            break;*/
    }

    _iterations_count = current_iteration;

    MemoryAPI::free_device_array(number_of_loops);
    MemoryAPI::free_device_array(incoming_degrees_without_loops);
    MemoryAPI::free_device_array(old_page_ranks);
    MemoryAPI::free_device_array(reversed_degrees);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template void page_rank_wrapper<int, float>(ExtendedCSRGraph<int, float> &_graph,
                                            float *_page_ranks,
                                            float _convergence_factor,
                                            int _max_iterations,
                                            int &_iterations_count,
                                            TraversalDirection _traversal_direction);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

