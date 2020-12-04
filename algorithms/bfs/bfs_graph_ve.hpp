#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

BFS_GraphVE::BFS_GraphVE(VectCSRGraph &_graph)
{
    GraphAbstractionsNEC graph_API(_graph);
    FrontierNEC frontier(_graph);

    ve_vertices_count = _graph.get_vertices_count(); // TODO only non-zero
    ve_edges_per_vertex = BFS_VE_SIZE;

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
    graph_API.scatter(_graph, frontier, copy_edge_to_ve); // TODO GATHER if directed
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

BFS_GraphVE::~BFS_GraphVE()
{
    MemoryAPI::free_array(ve_dst_ids);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

