#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
template <typename _T>
void PR::nec_page_rank(VectCSRGraph &_graph,
                       VerticesArray<_T> &_page_ranks,
                       _T _convergence_factor,
                       int _max_iterations)
{
    #ifdef __USE_MPI__
    vgl_library_data.set_data_exchange_policy(RECENTLY_CHANGED);
    #endif

    int vertices_count = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count();

    VGL_GRAPH_ABSTRACTIONS graph_API(_graph);
    VGL_FRONTIER frontier(_graph);

    VerticesArray<int> number_of_loops(_graph, GATHER);
    VerticesArray<int> incoming_degrees(_graph, GATHER);
    VerticesArray<int> incoming_degrees_without_loops(_graph, GATHER);
    VerticesArray<_T> reversed_degrees(_graph, GATHER);
    VerticesArray<_T> old_page_ranks(_graph, SCATTER);
    VerticesArray<VGL_PACK_TYPE> packed_data(_graph, SCATTER);

    #pragma omp parallel
    {};

    graph_API.change_traversal_direction(GATHER, frontier, number_of_loops, incoming_degrees,
                                         incoming_degrees_without_loops, reversed_degrees);

    auto get_incoming_degrees = [&incoming_degrees] __VGL_COMPUTE_ARGS__
    {
        incoming_degrees[src_id] = connections_count;
    };
    graph_API.compute(_graph, frontier, get_incoming_degrees);

    _T d = 0.85;
    _T k = (1.0 - d) / ((_T)vertices_count);

    auto init_data = [&_page_ranks, &number_of_loops, vertices_count] __VGL_COMPUTE_ARGS__
    {
        _page_ranks[src_id] = 1.0/vertices_count;
        number_of_loops[src_id] = 0;
    };
    graph_API.compute(_graph, frontier, init_data);

    auto calculate_number_of_loops = [&number_of_loops] __VGL_ADVANCE_ARGS__
    {
        if(src_id == dst_id)
        {
            number_of_loops[src_id] += 1;
        }
    };

    graph_API.gather(_graph, frontier, calculate_number_of_loops);

    #ifdef __USE_MPI__
    vgl_library_data.exchange_data(_graph, number_of_loops.get_ptr(), _graph.get_vertices_count(), GATHER);
    #endif

    auto calculate_degrees_without_loops = [incoming_degrees_without_loops, incoming_degrees, number_of_loops] __VGL_COMPUTE_ARGS__
    {
        incoming_degrees_without_loops[src_id] = incoming_degrees[src_id] - number_of_loops[src_id];
    };
    graph_API.compute(_graph, frontier, calculate_degrees_without_loops);

    auto calculate_reversed_degrees = [reversed_degrees, incoming_degrees_without_loops] __VGL_COMPUTE_ARGS__
    {
        reversed_degrees[src_id] = 1.0 / incoming_degrees_without_loops[src_id];
        if(incoming_degrees_without_loops[src_id] == 0)
            reversed_degrees[src_id] = 0;
    };
    graph_API.compute(_graph, frontier, calculate_reversed_degrees);

    graph_API.change_traversal_direction(SCATTER, frontier, old_page_ranks, reversed_degrees, _page_ranks, incoming_degrees_without_loops);

    Timer tm;
    tm.start();
    int iterations_count = 0;
    for(iterations_count = 0; iterations_count < _max_iterations; iterations_count++)
    {
        auto save_old_ranks = [old_page_ranks, _page_ranks] __VGL_COMPUTE_ARGS__
        {
            old_page_ranks[src_id] = _page_ranks[src_id];
            _page_ranks[src_id] = 0;
        };
        graph_API.compute(_graph, frontier, save_old_ranks);

        //graph_API.pack_vertices_arrays(packed_data, old_page_ranks, reversed_degrees);

        auto reduce_dangling_input = [incoming_degrees_without_loops, old_page_ranks, vertices_count]__VGL_REDUCE_ANY_ARGS__->_T
        {
            _T result = 0.0;
            if(incoming_degrees_without_loops[src_id] == 0)
            {
                result = old_page_ranks[src_id] / vertices_count;
            }
            return result;
        };
        _T dangling_input = graph_API.reduce<_T>(_graph, frontier, reduce_dangling_input, REDUCE_SUM);

        #pragma omp parallel
        {
            // todo cached loads

            auto edge_op = [_page_ranks, old_page_ranks, reversed_degrees, packed_data] __VGL_ADVANCE_ARGS__
            {
                _T dst_rank = old_page_ranks[dst_id];
                _T reversed_dst_links_num = reversed_degrees[dst_id];

                if(src_id != dst_id)
                    _page_ranks[src_id] += dst_rank * reversed_dst_links_num;
            };

            auto vertex_postprocess_op = [_page_ranks, k, d, dangling_input] __VGL_ADVANCE_POSTPROCESS_ARGS__
            {
                _page_ranks[src_id] = k + d * (_page_ranks[src_id] + dangling_input);
            };

            graph_API.scatter(_graph, frontier, edge_op, EMPTY_VERTEX_OP, vertex_postprocess_op, edge_op, EMPTY_VERTEX_OP, vertex_postprocess_op);
        };

        #ifdef __USE_MPI__
        vgl_library_data.exchange_data(_graph, _page_ranks.get_ptr(), _graph.get_vertices_count(), SCATTER);
        #endif

        auto reduce_ranks_sum = [_page_ranks]__VGL_REDUCE_ANY_ARGS__->_T
        {
            return _page_ranks[src_id];
        };
        _T ranks_sum = graph_API.reduce<_T>(_graph, frontier, reduce_ranks_sum, REDUCE_SUM);
        //cout << "ranks sum: " << ranks_sum << endl;
    }
    tm.end();


    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    performance_stats.print_algorithm_performance_stats("PR (Page Rank, NEC/multicore)", tm.get_time(), _graph.get_edges_count());
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
