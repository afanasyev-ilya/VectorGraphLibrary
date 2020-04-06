#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _TVertexValue, typename _TEdgeWeight>
void PR::nec_page_rank(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                   float *_page_ranks,
                   float _convergence_factor,
                   int _max_iterations)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    int   *number_of_loops;
    int   *incoming_degrees_without_loops;
    float *old_page_ranks;
    float *reversed_degrees;
    MemoryAPI::allocate_array(&number_of_loops, vertices_count);
    MemoryAPI::allocate_array(&incoming_degrees_without_loops, vertices_count);
    MemoryAPI::allocate_array(&old_page_ranks, vertices_count);
    MemoryAPI::allocate_array(&reversed_degrees, vertices_count);

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

    // TOFIX
    for(int src_id = 0; src_id < vertices_count; src_id++)
    {
        const long long edge_start = outgoing_ptrs[src_id];
        const int connections_count = outgoing_ptrs[src_id + 1] - outgoing_ptrs[src_id];

        for (int edge_pos = 0; edge_pos < connections_count; edge_pos++)
        {
            long long int global_edge_pos = edge_start + edge_pos;
            int dst_id = outgoing_ids[global_edge_pos];

            if(src_id == dst_id)
                number_of_loops[src_id]++;
        }
    }

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

        float dangling_input = 0;
        #pragma _NEC vector
        #pragma omp parallel for reduction(+: dangling_input)
        for(int i = 0; i < vertices_count; i++)
        {
            if(incoming_degrees_without_loops[i] == 0)
            {
                dangling_input += old_page_ranks[i] / vertices_count;
            }
        }

        auto vertex_preprocess_op = [](int src_id, int connections_count, int vector_index, DelayedWriteNEC &delayed_write)
        {
            #pragma _NEC vector
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                delayed_write.flt_vec_reg[i] = 0;
            }
        };

        auto edge_op = [_page_ranks, old_page_ranks, reversed_degrees](int src_id, int dst_id, int local_edge_pos,
                                                                             long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
        {
            float dst_rank = old_page_ranks[dst_id];
            float reversed_dst_links_num = reversed_degrees[dst_id];

            if(src_id != dst_id)
                delayed_write.flt_vec_reg[vector_index] += dst_rank * reversed_dst_links_num;
        };

        struct VertexPostprocessFunctor
        {
            float *page_ranks;
            float k;
            float d;
            float dangling_input;
            VertexPostprocessFunctor(float *_page_ranks, float _k, float _d, float _dangling_input): page_ranks(_page_ranks), k(_k), d(_d), dangling_input(_dangling_input) {}

            void operator()(int src_id, int connections_count, int vector_index, DelayedWriteNEC &delayed_write)
            {
                float new_rank = 0.0;
                #pragma _NEC unroll(VECTOR_LENGTH)
                for(int i = 0; i < VECTOR_LENGTH; i++)
                {
                    new_rank += delayed_write.flt_vec_reg[i];
                }
                page_ranks[src_id] = k + d * (new_rank + dangling_input);
                //page_ranks[src_id] = k + d * (page_ranks[src_id] + dangling_input);
            }
        };
        VertexPostprocessFunctor vertex_postprocess_op(_page_ranks, k, d, dangling_input);

        auto collective_vertex_preprocess_op = [](int src_id, int connections_count, int vector_index, DelayedWriteNEC &delayed_write)
        {
            delayed_write.flt_vec_reg[vector_index] = 0.0;
        };

        auto collective_vertex_postprocess_op = [_page_ranks, k, d, dangling_input](int src_id, int connections_count, int vector_index, DelayedWriteNEC &delayed_write)
        {
            _page_ranks[src_id] = k + d * (delayed_write.flt_vec_reg[vector_index] + dangling_input);
        };

        graph_API.advance(_graph, frontier, edge_op, vertex_preprocess_op, vertex_postprocess_op, edge_op, collective_vertex_preprocess_op, collective_vertex_postprocess_op);

        double ranks_sum = 0;
        #pragma omp parallel for reduction(+: ranks_sum)
        for(int i = 0; i < vertices_count; i++)
        {
            ranks_sum += _page_ranks[i];
        }
        cout << "ranks sum: " << ranks_sum << endl;

        if(fabs(ranks_sum - 1.0) > _convergence_factor)
        {
            cout << "ranks sum: " << ranks_sum << endl;
            throw "ERROR: page rank sum is incorrect";
        }
    }
    double t2 = omp_get_wtime();
    performance_stats("page ranks", t2 - t1, edges_count, iterations_count);

    MemoryAPI::free_array(number_of_loops);
    MemoryAPI::free_array(incoming_degrees_without_loops);
    MemoryAPI::free_array(old_page_ranks);
    MemoryAPI::free_array(reversed_degrees);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
