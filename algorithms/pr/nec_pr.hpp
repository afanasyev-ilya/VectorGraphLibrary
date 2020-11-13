#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _T>
void PR::nec_page_rank(VectCSRGraph &_graph,
                       VerticesArray<_T> &_page_ranks,
                       _T _convergence_factor,
                       int _max_iterations)
{
    int vertices_count = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count();
    GraphAbstractionsNEC graph_API(_graph, SCATTER);
    FrontierNEC frontier(_graph, SCATTER);

    VerticesArray<int> number_of_loops(_graph, SCATTER);
    VerticesArray<int> incoming_degrees(_graph, GATHER);
    VerticesArray<int> incoming_degrees_without_loops(_graph, SCATTER);
    VerticesArray<_T> old_page_ranks(_graph, SCATTER);
    VerticesArray<_T> reversed_degrees(_graph, SCATTER);
    VerticesArray<VGL_PACK_TYPE> packed_data(_graph, SCATTER);

    graph_API.change_traversal_direction(GATHER, frontier, incoming_degrees);

    auto get_incoming_degrees = [&incoming_degrees] (int src_id, int connections_count, int vector_index)
    {
        incoming_degrees[src_id] = connections_count;
    };
    graph_API.compute(_graph, frontier, get_incoming_degrees);

    graph_API.change_traversal_direction(SCATTER, frontier, incoming_degrees, number_of_loops, incoming_degrees_without_loops,
            old_page_ranks, reversed_degrees, _page_ranks);

    float d = 0.85;
    float k = (1.0 - d) / ((float)vertices_count);

    auto init_data = [&_page_ranks, &number_of_loops, vertices_count] (int src_id, int connections_count, int vector_index)
    {
        _page_ranks[src_id] = 1.0/vertices_count;
        number_of_loops[src_id] = 0;
    };
    graph_API.compute(_graph, frontier, init_data);

    auto calculate_number_of_loops = [&number_of_loops](int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos,
                    int vector_index, DelayedWriteNEC &delayed_write)
    {
        if(src_id == dst_id)
        {
            delayed_write.int_vec_reg[vector_index] += 1;
        }
    };

    auto calculate_number_of_loops_collective = [&number_of_loops](int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos,
                    int vector_index, DelayedWriteNEC &delayed_write)
    {
        if(src_id == dst_id)
        {
            number_of_loops[src_id] += 1;
        }
    };

    auto vertex_postprocess_calculate_number_of_loops = [&number_of_loops]
                        (int src_id, int connections_count, int vector_index, DelayedWriteNEC &delayed_write)
    {
        delayed_write.finish_write_sum(number_of_loops.get_ptr(), src_id);
    };

    _graph_API.enable_safe_stores();
    graph_API.scatter(_graph, frontier, calculate_number_of_loops, EMPTY_VERTEX_OP, vertex_postprocess_calculate_number_of_loops,
                      calculate_number_of_loops_collective, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);
    _graph_API.disable_safe_stores();

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

    Timer tm;
    tm.start();
    int iterations_count = 0;
    for(iterations_count = 0; iterations_count < _max_iterations; iterations_count++)
    {
        auto save_old_ranks = [old_page_ranks, _page_ranks] (int src_id, int connections_count, int vector_index)
        {
            old_page_ranks[src_id] = _page_ranks[src_id];
            _page_ranks[src_id] = 0;
        };
        graph_API.compute(_graph, frontier, save_old_ranks);

        graph_API.pack_vertices_arrays(packed_data, old_page_ranks, reversed_degrees);

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

        auto edge_op = [_page_ranks, old_page_ranks, reversed_degrees, packed_data](int src_id, int dst_id, int local_edge_pos,
                    long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
        {
            /*VGL_PACK_TYPE packed_val = packed_data[dst_id];

            delayed_write.pack_int_1[vector_index] = (int)((packed_val & 0xFFFFFFFF00000000LL) >> 32);
            delayed_write.pack_int_2[vector_index] = (int)(packed_val & 0xFFFFFFFFLL);

            float dst_rank = delayed_write.pack_int_1_to_flt[vector_index];
            float reversed_dst_links_num = delayed_write.pack_int_2_to_flt[vector_index];*/

            //if(dst_rank != old_page_ranks[dst_id])
            //    cout << dst_rank << " vs " << old_page_ranks[dst_id] << endl;

            float dst_rank = old_page_ranks[dst_id];
            float reversed_dst_links_num = reversed_degrees[dst_id];

            if(src_id != dst_id)
                _page_ranks[src_id] += dst_rank * reversed_dst_links_num;
        };

        auto vertex_postprocess_op = [_page_ranks, k, d, dangling_input](int src_id, int connections_count, int vector_index, DelayedWriteNEC &delayed_write)
        {
            _page_ranks[src_id] = k + d * (_page_ranks[src_id] + dangling_input);
        };

        graph_API.scatter(_graph, frontier, edge_op, EMPTY_VERTEX_OP, vertex_postprocess_op, edge_op, EMPTY_VERTEX_OP, vertex_postprocess_op);

        auto reduce_ranks_sum = [_page_ranks](int src_id, int connections_count, int vector_index)->float
        {
            return _page_ranks[src_id];
        };
        double ranks_sum = graph_API.reduce<double>(_graph, frontier, reduce_ranks_sum, REDUCE_SUM);
        cout << "ranks sum: " << ranks_sum << endl;
    }
    tm.end();

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    PerformanceStats::print_algorithm_performance_stats("page ranks", tm.get_time(), edges_count, iterations_count);
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
