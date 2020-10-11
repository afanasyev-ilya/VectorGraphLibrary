#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
void PR::nec_page_rank(UndirectedCSRGraph &_graph,
                       float *_page_ranks,
                       float _convergence_factor,
                       int _max_iterations)
{
    LOAD_UNDIRECTED_CSR_GRAPH_DATA(_graph);

    int   *number_of_loops;
    int   *incoming_degrees_without_loops;
    float *old_page_ranks;
    float *reversed_degrees;
    //uint64_t *packed_data;
    MemoryAPI::allocate_array(&number_of_loops, vertices_count);
    MemoryAPI::allocate_array(&incoming_degrees_without_loops, vertices_count);
    MemoryAPI::allocate_array(&old_page_ranks, vertices_count);
    MemoryAPI::allocate_array(&reversed_degrees, vertices_count);
    //MemoryAPI::allocate_array(&packed_data, vertices_count);

    float d = 0.85;
    float k = (1.0 - d) / ((float)vertices_count);

    GraphPrimitivesNEC graph_API;

    FrontierNEC frontier(vertices_count);
    frontier.set_all_active();

    auto init_data = [_page_ranks, number_of_loops, vertices_count] (int src_id, int connections_count, int vector_index)
    {
        _page_ranks[src_id] = 1.0/vertices_count;
        number_of_loops[src_id] = 0;
    };
    graph_API.compute(_graph, frontier, init_data);

    auto calculate_number_of_loops = [number_of_loops](int src_id, int dst_id, int local_edge_pos,
                    long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
    {
        if(src_id == dst_id)
        {
            delayed_write.int_vec_reg[vector_index] += 1;
        }
    };

    auto calculate_number_of_loops_collective = [number_of_loops](int src_id, int dst_id, int local_edge_pos,
                                     long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
    {
        if(src_id == dst_id)
        {
            number_of_loops[src_id] += 1;
        }
    };

    auto vertex_postprocess_calculate_number_of_loops = [number_of_loops]
                        (int src_id, int connections_count, int vector_index, DelayedWriteNEC &delayed_write)
    {
        delayed_write.finish_write_sum(number_of_loops, src_id);
    };

    graph_API.advance(_graph, frontier, calculate_number_of_loops, EMPTY_VERTEX_OP, vertex_postprocess_calculate_number_of_loops,
                      calculate_number_of_loops_collective, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);

    auto calculate_degrees_without_loops = [incoming_degrees_without_loops, incoming_degrees, number_of_loops] (int src_id, int connections_count, int vector_index)
    {
        incoming_degrees_without_loops[src_id] = incoming_degrees[src_id] - number_of_loops[src_id];
    };
    graph_API.compute(_graph, frontier, calculate_degrees_without_loops);

    auto calculate_reversed_degrees = [reversed_degrees, incoming_degrees_without_loops] (int src_id, int connections_count, int vector_index)
    {
        reversed_degrees[src_id] = 1.0 / incoming_degrees_without_loops[src_id];
    };
    graph_API.compute(_graph, frontier, calculate_reversed_degrees);

    double t1 = omp_get_wtime();
    int iterations_count = 0;
    for(iterations_count = 0; iterations_count < _max_iterations; iterations_count++)
    {
        auto save_old_ranks = [old_page_ranks, _page_ranks] (int src_id, int connections_count, int vector_index)
        {
            old_page_ranks[src_id] = _page_ranks[src_id];
            _page_ranks[src_id] = 0;
        };
        graph_API.compute(_graph, frontier, save_old_ranks);

        //pack_array_data(old_page_ranks, reversed_degrees, packed_data, vertices_count);

        auto reduce_dangling_input = [incoming_degrees_without_loops, old_page_ranks, vertices_count](int src_id, int connections_count, int vector_index)->float
        {
            float result = 0.0;
            if(incoming_degrees_without_loops[src_id] == 0)
            {
                result = old_page_ranks[src_id] / vertices_count;
            }
            return result;
        };
        double dangling_input = graph_API.reduce<double>(_graph, frontier, reduce_dangling_input, REDUCE_SUM);

        auto edge_op = [_page_ranks, old_page_ranks, reversed_degrees](int src_id, int dst_id, int local_edge_pos,
                                                                       long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
        {
            //uint64_t packed = packed_data[dst_id];
            float dst_rank = old_page_ranks[dst_id];
            float reversed_dst_links_num = reversed_degrees[dst_id];

            if(src_id != dst_id)
                _page_ranks[src_id] += dst_rank * reversed_dst_links_num;
        };

        auto vertex_postprocess_op = [_page_ranks, k, d, dangling_input](int src_id, int connections_count, int vector_index, DelayedWriteNEC &delayed_write)
        {
            _page_ranks[src_id] = k + d * (_page_ranks[src_id] + dangling_input);
        };

        graph_API.advance(_graph, frontier, edge_op, EMPTY_VERTEX_OP, vertex_postprocess_op);

        /*auto reduce_ranks_sum = [_page_ranks](int src_id, int connections_count, int vector_index)->float
        {
            return _page_ranks[src_id];
        };
        double ranks_sum = graph_API.reduce<double>(_graph, frontier, reduce_ranks_sum, REDUCE_SUM);*/
    }
    double t2 = omp_get_wtime();

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    PerformanceStats::print_performance_stats("page ranks", t2 - t1, edges_count, iterations_count);
    #endif

    MemoryAPI::free_array(number_of_loops);
    MemoryAPI::free_array(incoming_degrees_without_loops);
    MemoryAPI::free_array(old_page_ranks);
    MemoryAPI::free_array(reversed_degrees);
    //MemoryAPI::free_array(packed_data);

    performance_per_iteration = double(iterations_count) * (edges_count/((t2 - t1)*1e6));
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
