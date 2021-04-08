#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int UndirectedCSRGraph::get_edge_dst(int _src_id, int _edge_pos)
{
    return adjacent_ids[vertex_pointers[_src_id] + _edge_pos];
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int UndirectedCSRGraph::get_connections_count(int _vertex_id)
{
    return (vertex_pointers[_vertex_id + 1] - vertex_pointers[_vertex_id]);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

long long UndirectedCSRGraph::get_csr_edge_id(int _src_id, int _dst_id)
{
    const long long int start = vertex_pointers[_src_id];
    const long long int end = vertex_pointers[_src_id + 1];
    const int connections_count = end - start;

    for (int local_edge_pos = 0; local_edge_pos < connections_count; local_edge_pos++)
    {
        const long long global_edge_pos = start + local_edge_pos;
        const int dst_id = adjacent_ids[global_edge_pos];
        if(dst_id == _dst_id)
        {
            return global_edge_pos;
        }
    }
    throw "Error in UndirectedCSRGraph::get_csr_edge_id(): specified dst_id not found for current src vertex";
    return -1;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
