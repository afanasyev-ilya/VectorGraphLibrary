#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void PR::seq_page_rank(ExtendedCSRGraph &_graph,
                       float *_page_ranks,
                       float _convergence_factor,
                       int _max_iterations)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    // set PR parameters
    float d = 0.85;
    float k = (1.0 - d) / ((float)vertices_count);

    int   *number_of_loops;
    int   *incoming_degrees_without_loops;
    float *old_page_ranks;

    MemoryAPI::allocate_array(&number_of_loops, vertices_count);
    MemoryAPI::allocate_array(&incoming_degrees_without_loops, vertices_count);
    MemoryAPI::allocate_array(&old_page_ranks, vertices_count);

    // init ranks and other data
    for(int i = 0; i < vertices_count; i++)
    {
        _page_ranks[i] = 1.0/vertices_count;
        number_of_loops[i] = 0;
    }

    // calculate number of loops
    for(int src_id = 0; src_id < vertices_count; src_id++)
    {
        const long long edge_start = vertex_pointers[src_id];
        const int connections_count = vertex_pointers[src_id + 1] - vertex_pointers[src_id];

        for (int edge_pos = 0; edge_pos < connections_count; edge_pos++)
        {
            long long int global_edge_pos = edge_start + edge_pos;
            int dst_id = adjacent_ids[global_edge_pos];

            if(src_id == dst_id)
                number_of_loops[src_id]++;
        }
    }

    // calculate incoming degrees without loops
    for(int i = 0; i < vertices_count; i++)
    {
        incoming_degrees_without_loops[i] = incoming_degrees[i] - number_of_loops[i];
    }

    int iterations_count = 0;
    for(iterations_count = 0; iterations_count < _max_iterations; iterations_count++)
    {
        // copy ranks from prev iteration to temporary array
        for(int i = 0; i < vertices_count; i++)
        {
            old_page_ranks[i] = _page_ranks[i];
            _page_ranks[i] = 0;
        }

        // calculate dangling input
        float dangling_input = 0;
        for(int i = 0; i < vertices_count; i++)
        {
            if(incoming_degrees_without_loops[i] <= 0)
            {
                dangling_input += old_page_ranks[i] / vertices_count;
            }
        }

        // traverse graph and calculate page ranks
        for(int src_id = 0; src_id < vertices_count; src_id++)
        {
            const long long edge_start = vertex_pointers[src_id];
            const int connections_count = vertex_pointers[src_id + 1] - vertex_pointers[src_id];

            for (int edge_pos = 0; edge_pos < connections_count; edge_pos++)
            {
                long long int global_edge_pos = edge_start + edge_pos;
                int dst_id = adjacent_ids[global_edge_pos];

                float dst_rank = old_page_ranks[dst_id];
                float dst_links_num = 1.0 / incoming_degrees_without_loops[dst_id];

                if(src_id != dst_id)
                    _page_ranks[src_id] += dst_rank * dst_links_num;
            }

            _page_ranks[src_id] = k + d * (_page_ranks[src_id] + dangling_input);
        }

        // calculate new ranks sum
        double ranks_sum = 0;
        for(int i = 0; i < vertices_count; i++)
        {
            ranks_sum += _page_ranks[i];
        }

        /*if(fabs(ranks_sum - 1.0) > _convergence_factor)
        {
            cout << "ranks sum: " << ranks_sum << endl;
            throw "ERROR: page rank sum is incorrect";
        }*/
    }

    MemoryAPI::free_array(number_of_loops);
    MemoryAPI::free_array(incoming_degrees_without_loops);
    MemoryAPI::free_array(old_page_ranks);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
