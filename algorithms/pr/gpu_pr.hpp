#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _T>
void PR::gpu_page_rank(VectCSRGraph &_graph,
                       VerticesArray<_T> &_page_ranks,
                       _T _convergence_factor,
                       int _max_iterations)
{
    int vertices_count = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count();
    GraphAbstractionsGPU graph_API(_graph);
    FrontierGPU frontier(_graph);
    frontier.set_all_active();

    VerticesArray<int> number_of_loops(_graph, GATHER);
    VerticesArray<int> incoming_degrees(_graph, GATHER);
    VerticesArray<int> incoming_degrees_without_loops(_graph, GATHER);
    VerticesArray<_T> reversed_degrees(_graph, GATHER);
    VerticesArray<_T> old_page_ranks(_graph, SCATTER);

    graph_API.change_traversal_direction(GATHER, frontier, incoming_degrees, number_of_loops, incoming_degrees_without_loops, reversed_degrees);

    auto get_incoming_degrees = [incoming_degrees] __device__ (int src_id, int connections_count, int vector_index)
    {
        incoming_degrees[src_id] = connections_count;
    };
    graph_API.compute(_graph, frontier, get_incoming_degrees);

    float d = 0.85;
    float k = (1.0 - d) / ((float)vertices_count);

    auto init_data = [_page_ranks, number_of_loops, vertices_count] __device__ (int src_id, int connections_count, int vector_index)
    {
        _page_ranks[src_id] = 1.0/vertices_count;
        number_of_loops[src_id] = 0;
    };
    graph_API.compute(_graph, frontier, init_data);

    auto calculate_number_of_loops = [number_of_loops] __device__ (int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos,
                    int vector_index)
    {
        if(src_id == dst_id)
        {
            atomicAdd(&number_of_loops[src_id], 1);
        }
    };

    graph_API.gather(_graph, frontier, calculate_number_of_loops);

    auto calculate_degrees_without_loops = [incoming_degrees_without_loops, incoming_degrees, number_of_loops] __device__ (int src_id, int connections_count, int vector_index)
    {
        incoming_degrees_without_loops[src_id] = incoming_degrees[src_id] - number_of_loops[src_id];
    };
    graph_API.compute(_graph, frontier, calculate_degrees_without_loops);

    auto calculate_reversed_degrees = [reversed_degrees, incoming_degrees_without_loops] __device__ (int src_id, int connections_count, int vector_index)
    {
        reversed_degrees[src_id] = 1.0 / incoming_degrees_without_loops[src_id];
        if(incoming_degrees_without_loops[src_id] == 0)
            reversed_degrees[src_id] = 0;
    };
    graph_API.compute(_graph, frontier, calculate_reversed_degrees);

    graph_API.change_traversal_direction(SCATTER, frontier, old_page_ranks, reversed_degrees, _page_ranks, incoming_degrees_without_loops);

    incoming_degrees_without_loops.reorder(SCATTER);
    reversed_degrees.reorder(SCATTER);

    Timer tm;
    tm.start();
    int iterations_count = 0;
    frontier.set_all_active();
    for(iterations_count = 0; iterations_count < _max_iterations; iterations_count++)
    {
        auto save_old_ranks = [old_page_ranks, _page_ranks] __device__ (int src_id, int connections_count, int vector_index)
        {
            old_page_ranks[src_id] = _page_ranks[src_id];
            _page_ranks[src_id] = 0;
        };
        graph_API.compute(_graph, frontier, save_old_ranks);

        auto reduce_dangling_input = [incoming_degrees_without_loops, old_page_ranks, vertices_count] __device__ (int src_id, int connections_count, int vector_index)->float
        {
            float result = 0.0;
            if(incoming_degrees_without_loops[src_id] == 0)
            {
                result = old_page_ranks[src_id] / vertices_count;
            }
            return result;
        };
        double dangling_input = graph_API.reduce<double>(_graph, frontier, reduce_dangling_input, REDUCE_SUM);

        auto edge_op = [_page_ranks, old_page_ranks, reversed_degrees, incoming_degrees_without_loops] __device__ (int src_id, int dst_id, int local_edge_pos,
                    long long int global_edge_pos, int vector_index)
        {
            float dst_rank = old_page_ranks[dst_id];
            float reversed_dst_links_num = reversed_degrees[dst_id];

            if(src_id != dst_id)
                atomicAdd(&_page_ranks[src_id], dst_rank * reversed_dst_links_num);
        };
        /*auto EMPTY_VERTEX_OP = [] __device__(int src_id, int position_in_frontier, int connections_count){};

        auto vertex_postprocess_op = [_page_ranks, k, d, dangling_input] __device__ (int src_id, int connections_count, int vector_index)
        {
            _page_ranks[src_id] = k + d * (_page_ranks[src_id] + dangling_input);
        };

        graph_API.scatter(_graph, frontier, edge_op, EMPTY_VERTEX_OP, vertex_postprocess_op, edge_op, EMPTY_VERTEX_OP, vertex_postprocess_op);*/

        graph_API.scatter(_graph, frontier, edge_op);

        auto save_ranks = [_page_ranks, k, d, dangling_input] __device__ (int src_id, int connections_count, int vector_index)
        {
            _page_ranks[src_id] = k + d * (_page_ranks[src_id] + dangling_input);
        };
        graph_API.compute(_graph, frontier, save_ranks);

        auto reduce_ranks_sum = [_page_ranks] __device__ (int src_id, int connections_count, int vector_index)->float
        {
            return _page_ranks[src_id];
        };
        double ranks_sum = graph_API.reduce<double>(_graph, frontier, reduce_ranks_sum, REDUCE_SUM);
        cout << "ranks sum: " << ranks_sum << endl;
    }
    tm.end();

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    PerformanceStats::print_algorithm_performance_stats("PR (Page Rank, GPU)", tm.get_time(), edges_count, iterations_count);
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
