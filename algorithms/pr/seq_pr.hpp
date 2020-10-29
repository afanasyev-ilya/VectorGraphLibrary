#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void PR::seq_page_rank(VectCSRGraph &_graph,
                       VerticesArray<_T> &_page_ranks,
                       _T _convergence_factor,
                       int _max_iterations)
{
    // get graph pointers
    UndirectedCSRGraph *outgoing_graph_ptr = _graph.get_outgoing_graph_ptr();
    UndirectedCSRGraph *incoming_graph_ptr = _graph.get_outgoing_graph_ptr();
    int vertices_count = _graph.get_vertices_count();

    // set PR parameters
    _T d = 0.85;
    _T k = (1.0 - d) / ((float)vertices_count);

    VerticesArray<int> number_of_loops(_graph, SCATTER);
    VerticesArray<int> incoming_degrees(_graph, GATHER);
    VerticesArray<int> incoming_degrees_without_loops(_graph, SCATTER);
    VerticesArray<_T> reversed_incoming_degrees(_graph, SCATTER);
    VerticesArray<_T> old_page_ranks(_graph, SCATTER);

    // init ranks and other data
    for(int i = 0; i < vertices_count; i++)
    {
        _page_ranks[i] = 1.0/vertices_count;
        number_of_loops[i] = 0;
    }

    // calculate number of loops
    for(int src_id = 0; src_id < vertices_count; src_id++)
    {
        const long long first_edge = outgoing_graph_ptr->get_vertex_pointers()[src_id];
        const long long last_edge = outgoing_graph_ptr->get_vertex_pointers()[src_id + 1];
        const int connections_count = last_edge - first_edge;

        for (int edge_pos = 0; edge_pos < connections_count; edge_pos++)
        {
            long long int global_edge_pos = first_edge + edge_pos;
            int dst_id = outgoing_graph_ptr->get_adjacent_ids()[global_edge_pos];

            if(src_id == dst_id)
                number_of_loops[src_id]++;
        }
    }

    // calculate incoming degrees without loops
    for(int i = 0; i < vertices_count; i++)
    {
        const long long first_edge = incoming_graph_ptr->get_vertex_pointers()[i];
        const long long last_edge = incoming_graph_ptr->get_vertex_pointers()[i + 1];
        incoming_degrees[i] = last_edge - first_edge;
    }
    incoming_degrees.reorder(SCATTER);

    // calculate incoming degrees without loops
    for(int i = 0; i < vertices_count; i++)
    {
        incoming_degrees_without_loops[i] = incoming_degrees[i] - number_of_loops[i];
        reversed_incoming_degrees[i] = 1.0 / incoming_degrees_without_loops[i];
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
            const long long first_edge = outgoing_graph_ptr->get_vertex_pointers()[src_id];
            const long long last_edge = outgoing_graph_ptr->get_vertex_pointers()[src_id + 1];
            const int connections_count = last_edge - first_edge;

            for (int edge_pos = 0; edge_pos < connections_count; edge_pos++)
            {
                long long int global_edge_pos = first_edge + edge_pos;
                int dst_id = outgoing_graph_ptr->get_adjacent_ids()[global_edge_pos];

                float dst_rank = old_page_ranks[dst_id];
                float dst_links_num = reversed_incoming_degrees[dst_id];

                if(src_id != dst_id)
                    _page_ranks[src_id] += dst_rank * dst_links_num;
            }

            _page_ranks[src_id] = k + d * (_page_ranks[src_id] + dangling_input);
        }

        for(int i = 0; i < vertices_count; i++)
            cout << incoming_degrees_without_loops[i] << " ";
        cout << endl;

        for(int i = 0; i < vertices_count; i++)
            cout << _page_ranks[i] << " ";
        cout << endl;

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
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
