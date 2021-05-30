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

inline bool in_between(size_t _val, size_t _first, size_t _second)
{
    if((_first <= _val) && (_val < _second))
        return true;
    return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int UndirectedCSRGraph::get_non_zero_degree_vertices_count()
{
    int result = this->vertices_count;
    #pragma omp parallel for
    for(int src_id = 0; src_id < this->vertices_count - 1; src_id++)
    {
        int current_connections = vertex_pointers[src_id + 1] - vertex_pointers[src_id];
        int next_connections = vertex_pointers[src_id + 2] - vertex_pointers[src_id + 1];
        if(current_connections > 0 && next_connections == 0)
            result = src_id + 1;
    }
    return result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_MPI__
std::pair<int, int> UndirectedCSRGraph::get_mpi_thresholds(int _mpi_rank, int _v1, int _v2)
{
    size_t edges_counter = 0;
    int mpi_proc_num = vgl_library_data.get_mpi_proc_num();

    size_t group_edges_count = vertex_pointers[_v2] - vertex_pointers[_v1];
    size_t edges_per_mpi_proc = group_edges_count / mpi_proc_num;

    int first_vertex = _v1;
    int last_vertex = _v2;

    size_t first_edge_border = _mpi_rank * edges_per_mpi_proc + vertex_pointers[_v1];
    size_t last_edge_border = (_mpi_rank + 1) * edges_per_mpi_proc + vertex_pointers[_v1];

    #pragma _NEC ivdep
    #pragma omp parallel for
    for(int _vertex_id = _v1; _vertex_id < _v2 - 1; _vertex_id++)
    {
        size_t current = vertex_pointers[_vertex_id];
        size_t next = vertex_pointers[_vertex_id + 1];
        if(in_between(first_edge_border, current, next))
            first_vertex = _vertex_id;
        if(in_between(last_edge_border, current, next))
            last_vertex = _vertex_id;
    }

    //cout << vgl_library_data.get_mpi_rank() << " _ (" << _v1 << " " << _v2 << ") " << first_vertex << " " << last_vertex << endl;
    return make_pair(first_vertex, last_vertex);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
