#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

BFS_GraphVE::BFS_GraphVE(VectCSRGraph &_graph)
{
    #ifdef __USE_NEC_SX_AURORA__
    GraphAbstractionsNEC graph_API(_graph, SCATTER);
    FrontierNEC frontier(_graph, SCATTER);
    //GraphAbstractionsNEC graph_API(_graph, GATHER);
    //FrontierNEC frontier(_graph, GATHER);

    _graph.get_incoming_graph_ptr()->sort_adjacent_edges();

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
    ve_edges_per_vertex = BFS_VE_SIZE;

    cout << "non_zero_count: " << non_zero_count << endl;

    MemoryAPI::allocate_array(&ve_dst_ids, ve_vertices_count * ve_edges_per_vertex);
    #pragma _NEC vector
    #pragma omp parallel
    for(int i = 0; i < ve_vertices_count * ve_edges_per_vertex; i++)
        ve_dst_ids[i] = -1;

    // copy into local variables
    int l_ve_vertices_count = ve_vertices_count;
    int l_ve_edges_per_vertex = ve_edges_per_vertex;
    int *l_ve_dst_ids = ve_dst_ids;
    auto copy_edge_to_ve = [l_ve_vertices_count,l_ve_edges_per_vertex,l_ve_dst_ids](int src_id, int dst_id, int local_edge_pos,
                              long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
    {
        if(local_edge_pos < l_ve_edges_per_vertex)
            l_ve_dst_ids[src_id + l_ve_vertices_count*local_edge_pos] = dst_id;
    };
    frontier.set_all_active();
    //graph_API.gather(_graph, frontier, copy_edge_to_ve); // TODO GATHER if directed
    graph_API.scatter(_graph, frontier, copy_edge_to_ve);
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

BFS_GraphVE::~BFS_GraphVE()
{
    MemoryAPI::free_array(ve_dst_ids);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

