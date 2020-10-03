#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
VectorExtension<_TVertexValue, _TEdgeWeight>::VectorExtension()
{
    vertices_count = VECTOR_LENGTH;
    starting_vertex = 0;
    vector_segments_count = 1;

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

    free();
    alloc(edges_count + VECTOR_LENGTH);

    long long current_edge = 0;
    for(int cur_vector_segment = 0; cur_vector_segment < vector_segments_count; cur_vector_segment++)
    {
        int vec_start = cur_vector_segment * VECTOR_LENGTH + starting_vertex;
        int cur_max_connections_count = _csr_adjacent_ptrs[vec_start + 1] - _csr_adjacent_ptrs[vec_start];

        vector_group_ptrs[cur_vector_segment] = current_edge;
        vector_group_sizes[cur_vector_segment] = cur_max_connections_count;

        for(int edge_pos = 0; edge_pos < cur_max_connections_count; edge_pos++)
        {
            #pragma unroll
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
