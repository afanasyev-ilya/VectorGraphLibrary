#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
VectorExtension<_TVertexValue, _TEdgeWeight>::VectorExtension()
{
    vertices_count = VECTOR_LENGTH;
    starting_vertex = 0;
    vector_segments_count = 1;

    edges_count_in_ve = VECTOR_LENGTH;
    alloc(VECTOR_LENGTH);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
VectorExtension<_TVertexValue, _TEdgeWeight>::~VectorExtension()
{
    free();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void VectorExtension<_TVertexValue, _TEdgeWeight>::alloc(long long _edges_count)
{
    MemoryAPI::allocate_array(&vector_group_ptrs, vector_segments_count);
    MemoryAPI::allocate_array(&vector_group_sizes, vector_segments_count);

    MemoryAPI::allocate_array(&adjacent_ids, _edges_count);
    MemoryAPI::allocate_array(&adjacent_weights, _edges_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void VectorExtension<_TVertexValue, _TEdgeWeight>::free()
{
    MemoryAPI::free_array(vector_group_ptrs);
    MemoryAPI::free_array(vector_group_sizes);

    MemoryAPI::free_array(adjacent_ids);
    MemoryAPI::free_array(adjacent_weights);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void VectorExtension<_TVertexValue, _TEdgeWeight>::init_from_graph(long long *_csr_adjacent_ptrs,
                                                                   int *_csr_adjacent_ids,
                                                                   _TEdgeWeight *_csr_adjacent_weights,
                                                                   int _first_vertex,
                                                                   int _last_vertex)
{
    double t1 = omp_get_wtime();
    vertices_count = _last_vertex - _first_vertex;
    starting_vertex = _first_vertex;
    vector_segments_count = (vertices_count - 1) / VECTOR_LENGTH + 1;

    long long edges_count = 0;
    for(int cur_vector_segment = 0; cur_vector_segment < vector_segments_count; cur_vector_segment++)
    {
        int vec_start = cur_vector_segment * VECTOR_LENGTH + starting_vertex;
        int cur_max_connections_count = _csr_adjacent_ptrs[vec_start + 1] - _csr_adjacent_ptrs[vec_start];
        edges_count += cur_max_connections_count * VECTOR_LENGTH;
    }
    edges_count_in_ve = edges_count;

    free();
    alloc(edges_count + VECTOR_LENGTH);

    this->csr_adjacent_ptrs_ptr = _csr_adjacent_ptrs;
    this->first_vertex = _first_vertex;
    this->last_vertex = _last_vertex;

    long long current_edge = 0;
    for(int cur_vector_segment = 0; cur_vector_segment < vector_segments_count; cur_vector_segment++)
    {
        int vec_start = cur_vector_segment * VECTOR_LENGTH + starting_vertex;
        int cur_max_connections_count = _csr_adjacent_ptrs[vec_start + 1] - _csr_adjacent_ptrs[vec_start];

        vector_group_ptrs[cur_vector_segment] = current_edge;
        vector_group_sizes[cur_vector_segment] = cur_max_connections_count;

        for(int edge_pos = 0; edge_pos < cur_max_connections_count; edge_pos++)
        {
            #pragma _NEC ivdep
            #pragma _NEC vector
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                int src_id = vec_start + i;
                int connections_count = _csr_adjacent_ptrs[src_id + 1] - _csr_adjacent_ptrs[src_id];
                long long global_edge_pos = _csr_adjacent_ptrs[src_id] + edge_pos;
                if((src_id < _last_vertex) && (edge_pos < connections_count))
                {
                    adjacent_ids[current_edge + i] = _csr_adjacent_ids[global_edge_pos];
                    adjacent_weights[current_edge + i] = _csr_adjacent_weights[global_edge_pos];
                }
                else
                {
                    adjacent_ids[current_edge + i] = src_id;
                    adjacent_weights[current_edge + i] = 0.0;
                }
            }
            current_edge += VECTOR_LENGTH;
        }
    }
    double t2 = omp_get_wtime();
    cout << "VE creation time: " << t2 - t1 << " sec" << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
template <typename _T>
void VectorExtension<_TVertexValue, _TEdgeWeight>::copy_array_from_csr_to_ve(_T *_dst_ve_array, _T *_src_csr_array)
{
    #pragma omp parallel for
    for(int cur_vector_segment = 0; cur_vector_segment < vector_segments_count; cur_vector_segment++)
    {
        int vec_start = cur_vector_segment * VECTOR_LENGTH + first_vertex;
        long long edge_ptr = vector_group_ptrs[cur_vector_segment];
        int cur_max_connections_count = vector_group_sizes[cur_vector_segment];

        #pragma _NEC novector
        for(int edge_pos = 0; edge_pos < cur_max_connections_count; edge_pos++)
        {
            #pragma _NEC ivdep
            #pragma _NEC vector
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                int src_id = vec_start + i;
                int connections_count = csr_adjacent_ptrs_ptr[src_id + 1] - csr_adjacent_ptrs_ptr[src_id];
                long long csr_edge_pos = csr_adjacent_ptrs_ptr[src_id] + edge_pos;

                if((src_id < last_vertex) && (edge_pos < connections_count))
                {
                    _dst_ve_array[edge_ptr + edge_pos*VECTOR_LENGTH + i] = _src_csr_array[csr_edge_pos];
                }
                else
                {
                    _dst_ve_array[edge_ptr + edge_pos*VECTOR_LENGTH + i] = 0.0;
                }
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
inline long long VectorExtension<_TVertexValue, _TEdgeWeight>::get_ve_edge_id(int _src_id, int _dst_id)
{
    int cur_vector_segment = (_src_id - first_vertex)/VECTOR_LENGTH;
    int vec_start = cur_vector_segment * VECTOR_LENGTH + first_vertex;

    long long edge_ptr = vector_group_ptrs[cur_vector_segment];
    int cur_max_connections_count = vector_group_sizes[cur_vector_segment];

    #pragma _NEC novector
    for(int edge_pos = 0; edge_pos < cur_max_connections_count; edge_pos++)
    {
        #pragma _NEC novector
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            int src_id = vec_start + i;
            if(src_id == _src_id)
            {
                int dst_id = adjacent_ids[edge_ptr + edge_pos*VECTOR_LENGTH + i];
                if(dst_id == _dst_id)
                    return edge_ptr + edge_pos*VECTOR_LENGTH + i;
            }
        }
    }
    throw "Error in VectorExtension::get_csr_edge_id(): specified dst_id not found for current src vertex";
    return -1;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

