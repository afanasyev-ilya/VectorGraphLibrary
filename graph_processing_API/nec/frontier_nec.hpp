#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FrontierNEC::FrontierNEC(int _vertices_count)
{
    max_frontier_size = _vertices_count;
    current_frontier_size = 0;
    frontier_ids = new int[max_frontier_size];
    frontier_flags = new int[max_frontier_size];
    work_buffer = new int[max_frontier_size];

    #pragma omp parallel for schedule(static)
    for(int i = 0; i < max_frontier_size; i++)
    {
        frontier_ids[i] = 0;
        work_buffer[i] = 0;
        frontier_flags[i] = NEC_NOT_IN_FRONTIER_FLAG;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FrontierNEC::~FrontierNEC()
{
    delete []frontier_ids;
    delete []frontier_flags;
    delete []work_buffer;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int FrontierNEC::get_non_zero_value(int _private_value)
{
    #pragma omp barrier
    shared_tmp_val_int = -1;
    #pragma omp barrier

    #pragma omp critical
    {
        if (_private_value >= shared_tmp_val_int)
        {
            shared_tmp_val_int = _private_value;
        }
    }
    #pragma omp barrier
    _private_value = shared_tmp_val_int;
    #pragma omp barrier

    return _private_value;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierNEC::set_all_active()
{
    current_frontier_size = max_frontier_size;

    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < max_frontier_size; i++)
    {
        frontier_ids[i] = i;
        frontier_flags[i] = NEC_IN_FRONTIER_FLAG;
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierNEC::split_sorted_frontier(const long long *_vertex_pointers,
                                        int &_large_threshold_start,
                                        int &_large_threshold_end,
                                        int &_medium_threshold_start,
                                        int &_medium_threshold_end,
                                        int &_small_threshold_start,
                                        int &_small_threshold_end)
{
    int large_threshold_size = VECTOR_LENGTH*MAX_SX_AURORA_THREADS*16;
    int medium_threshold_size = VECTOR_LENGTH;

    int large_threshold_vertex = -1;
    int medium_threshold_vertex = -1;

    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma omp for schedule(static)
    for(int idx = 0; idx < current_frontier_size; idx++)
    {
        const int current_id = frontier_ids[idx];
        const int current_size = _vertex_pointers[current_id + 1] - _vertex_pointers[current_id];

        int next_id = 0;
        int next_size = 0;
        if(idx == (current_frontier_size - 1))
        {
            next_id = -1;
            next_size = 0;
        }
        else
        {
            next_id = frontier_ids[idx + 1];
            next_size = _vertex_pointers[next_id + 1] - _vertex_pointers[next_id];
        }

        if((current_size >= large_threshold_size) && (next_size < large_threshold_size))
        {
            large_threshold_vertex = idx + 1;
        }
        else if((current_size >= medium_threshold_size) && (next_size < medium_threshold_size))
        {
            medium_threshold_vertex = idx + 1;
        }
    }

    #pragma omp barrier

    large_threshold_vertex = get_non_zero_value(large_threshold_vertex);
    medium_threshold_vertex = get_non_zero_value(medium_threshold_vertex);

    #pragma omp barrier

    if(large_threshold_vertex == -1)
    {
        large_threshold_vertex = 0;
    }
    if(medium_threshold_vertex == -1)
    {
        medium_threshold_vertex = large_threshold_vertex;
    }

    _large_threshold_start = 0;
    _large_threshold_end = large_threshold_vertex;
    _medium_threshold_start = large_threshold_vertex;
    _medium_threshold_end = medium_threshold_vertex;
    _small_threshold_start = medium_threshold_vertex;
    _small_threshold_end = current_frontier_size;

    #pragma omp barrier
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight, typename Condition>
void FrontierNEC::generate_frontier(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                    Condition condition_op)
{
    int vertices_count = _graph.get_vertices_count();

    current_frontier_size = dense_copy_if(frontier_ids, vertices_count, condition_op);

    /*cout << "front size: " << current_frontier_size << endl;

    for(int i = 0; i < min(current_frontier_size, 40); i++)
        cout << frontier_ids[i] << "(" << _graph.get_outgoing_ptrs()[frontier_ids[i]+1]-_graph.get_outgoing_ptrs()[frontier_ids[i]] << ") ";
    cout << endl;*/
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Condition>
void FrontierNEC::set_frontier_flags(Condition condition_op)
{
    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < max_frontier_size; i++)
    {
        if(condition_op(i) == true)
        {
            frontier_flags[i] = NEC_IN_FRONTIER_FLAG;
        }
        else
        {
            frontier_flags[i] = NEC_NOT_IN_FRONTIER_FLAG;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

