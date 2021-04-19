#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

BFS_GraphVE::BFS_GraphVE(VectCSRGraph &_graph)
{
    /*#ifdef __USE_NEC_SX_AURORA__
    GraphAbstractionsNEC graph_API(_graph, SCATTER);
    FrontierNEC frontier(_graph, SCATTER);
    //GraphAbstractionsNEC graph_API(_graph, GATHER);
    //FrontierNEC frontier(_graph, GATHER);

    // already should be sorted
    //_graph.get_incoming_graph_ptr()->sort_adjacent_edges();

    frontier.set_all_active();
    auto calculate_non_zero_count = []__VGL_COMPUTE_ARGS__->int
    {
        int result = 0;
        if(connections_count > 0)
        {
            result = 1;
        }
        return result;
    };
    int non_zero_count = graph_API.reduce<int>(_graph, frontier, calculate_non_zero_count, REDUCE_SUM);

    while(non_zero_count % VECTOR_LENGTH != 0)
    {
        non_zero_count++;
    }
    ve_vertices_count = non_zero_count; // TODO only non-zero

    cout << "non_zero_count: " << non_zero_count << endl;

    MemoryAPI::allocate_array(&ve_dst_ids, ve_vertices_count * BFS_VE_SIZE);
    #pragma _NEC vector
    #pragma omp parallel
    for(int i = 0; i < ve_vertices_count * BFS_VE_SIZE; i++)
        ve_dst_ids[i] = -1;


    int l_ve_vertices_count = ve_vertices_count;
    int l_ve_edges_per_vertex = ve_edges_per_vertex;
    int *l_ve_dst_ids = ve_dst_ids;
    auto copy_edge_to_ve = [l_ve_vertices_count,l_ve_edges_per_vertex,l_ve_dst_ids](int src_id, int dst_id, int local_edge_pos,
                              long long int global_edge_pos, int vector_index)
    {
        int prev_segments = (src_id - (src_id % VECTOR_LENGTH))/VECTOR_LENGTH;
        long long ve_pos = VECTOR_LENGTH * BFS_VE_SIZE * prev_segments + local_edge_pos * VECTOR_LENGTH + vector_index;

        if(local_edge_pos < l_ve_edges_per_vertex)
            l_ve_dst_ids[src_id + l_ve_vertices_count*local_edge_pos] = dst_id;
    };
    frontier.set_all_active();
    graph_API.scatter(_graph, frontier, copy_edge_to_ve);

    // copy into local variables
    int l_ve_vertices_count = ve_vertices_count;
    int l_ve_edges_per_vertex = ve_edges_per_vertex;
    int *l_ve_dst_ids = ve_dst_ids;
    auto copy_edge_to_ve = [l_ve_vertices_count,l_ve_edges_per_vertex,l_ve_dst_ids](int src_id, int dst_id, int local_edge_pos,
                              long long int global_edge_pos, int vector_index)
    {
        if(local_edge_pos < l_ve_edges_per_vertex)
            l_ve_dst_ids[src_id + l_ve_vertices_count*local_edge_pos] = dst_id;
    };
    frontier.set_all_active();
    //graph_API.gather(_graph, frontier, copy_edge_to_ve); // TODO GATHER if directed
    graph_API.scatter(_graph, frontier, copy_edge_to_ve);
    #endif*/

    int vertices_count       = _graph.get_vertices_count();
    long long *outgoing_ptrs = _graph.get_outgoing_graph_ptr()->get_vertex_pointers();
    int       *outgoing_ids  = _graph.get_outgoing_graph_ptr()->get_adjacent_ids();

    int zero_nodes_count = 0;
    #pragma _NEC vector
    #pragma omp parallel for schedule(static) reduction(+: zero_nodes_count)
    for(int src_id = 0; src_id < vertices_count; src_id++)
    {
        int connections = outgoing_ptrs[src_id + 1] - outgoing_ptrs[src_id];
        if(connections == 0)
        {
            zero_nodes_count++;
        }
    }
    int non_zero_vertices_count = vertices_count - zero_nodes_count;

    ve_vertices_count = non_zero_vertices_count;
    ve_edges_per_vertex = BOTTOM_UP_THRESHOLD;
    MemoryAPI::allocate_array(&ve_dst_ids, non_zero_vertices_count * BOTTOM_UP_THRESHOLD);

    for(int step = 0; step < BOTTOM_UP_THRESHOLD; step++)
    {
        #pragma omp parallel for schedule(static)
        for(int src_id = 0; src_id < non_zero_vertices_count; src_id++)
        {
            int connections = outgoing_ptrs[src_id + 1] - outgoing_ptrs[src_id];
            long long start_pos = outgoing_ptrs[src_id];

            if(step < connections)
            {
                int shift = step;
                int dst_id = outgoing_ids[start_pos + shift];
                ve_dst_ids[src_id + non_zero_vertices_count * step] = dst_id;
            }
        }
    }

    #pragma omp parallel
    {}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

BFS_GraphVE::~BFS_GraphVE()
{
    MemoryAPI::free_array(ve_dst_ids);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

