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

#ifdef __USE_MPI__
std::pair<int, int> UndirectedCSRGraph::get_mpi_thresholds(int _mpi_rank, TraversalDirection _direction)
{
    size_t edges_counter = 0;
    int mpi_proc_num = 1;
    MPI_Comm_size (MPI_COMM_WORLD, &mpi_proc_num); // TODO get from API

    size_t edges_per_mpi_proc = edges_count / mpi_proc_num;

    int first_vertex = 0;
    int last_vertex = this->vertices_count;

    size_t first_edge_border = _mpi_rank * edges_per_mpi_proc;
    size_t last_edge_border = (_mpi_rank + 1) * edges_per_mpi_proc;

    #pragma _NEC ivdep
    #pragma omp parallel for
    for(int _vertex_id = 0; _vertex_id < this->vertices_count - 1; _vertex_id++)
    {
        size_t current = vertex_pointers[_vertex_id];
        size_t next = vertex_pointers[_vertex_id + 1];
        if(in_between(first_edge_border, current, next)) // TODO check problem with border
            first_vertex = _vertex_id;
        if(in_between(last_edge_border, current, next))
            last_vertex = _vertex_id;
    }
    return make_pair(first_vertex, last_vertex);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
