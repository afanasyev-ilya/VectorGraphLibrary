#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void collect_number_of_accesses_per_vertex(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                           vector<int> &_dst_ptrs_data)
{
    LOAD_VECTORISED_CSR_GRAPH_REVERSE_DATA(_graph)
    
    for(int i = 0; i < vertices_count; i++)
    {
        _dst_ptrs_data[i] = 0;
    }
    
    for(int src_id = 0; src_id < number_of_vertices_in_first_part; src_id++)
    {
        long long edge_start = first_part_ptrs[src_id];
        int connections_count = first_part_sizes[src_id];
        
        for(int edge_pos = 0; edge_pos < connections_count; edge_pos++)
        {
            int dst_id = incoming_ids[edge_start + edge_pos];
            _dst_ptrs_data[dst_id]++;
        }
    }
    
    for(int cur_vector_segment = 0; cur_vector_segment < vector_segments_count; cur_vector_segment++)
    {
        int segment_first_vertex = cur_vector_segment * VECTOR_LENGTH + number_of_vertices_in_first_part;
        
        long long segement_edges_start = vector_group_ptrs[cur_vector_segment];
        int segment_connections_count = vector_group_sizes[cur_vector_segment];
        
        for(int edge_pos = 0; edge_pos < segment_connections_count; edge_pos++)
        {
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                int dst_id = incoming_ids[segement_edges_start + edge_pos * VECTOR_LENGTH + i];
                _dst_ptrs_data[dst_id]++;
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void create_access_profiling_file(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                  string _output_file_name)
{
    cout << "creating access profiling file: " << _output_file_name.c_str() << endl;
    
    vector<int> dst_ptrs_data(_graph.get_vertices_count());
    collect_number_of_accesses_per_vertex(_graph, dst_ptrs_data);
    
    cout << "data collected...." << endl;
    
    ofstream output_file;
    output_file.open(_output_file_name.c_str());
    for(int i = 0; i < _graph.get_vertices_count(); i++)
    {
        output_file << dst_ptrs_data[i] << " ";
    }
    output_file.close();
    cout << "saved!" << endl << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void collect_number_of_connections(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                   string _output_file_name)
{
    LOAD_VECTORISED_CSR_GRAPH_DATA(_graph)
    
    vector<int> connections_data(vertices_count);
    
    for(int src_id = 0; src_id < number_of_vertices_in_first_part; src_id++)
    {
        long long edge_start = first_part_ptrs[src_id];
        int connections_count = first_part_sizes[src_id];
        
        connections_data[src_id] = connections_count;
    }
    
    for(int cur_vector_segment = 0; cur_vector_segment < vector_segments_count; cur_vector_segment++)
    {
        int segment_first_vertex = cur_vector_segment * VECTOR_LENGTH + number_of_vertices_in_first_part;
        
        long long segement_edges_start = vector_group_ptrs[cur_vector_segment];
        int segment_connections_count = vector_group_sizes[cur_vector_segment];
        
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            connections_data[cur_vector_segment * VECTOR_LENGTH + i] = segment_connections_count;
        }
    }
    
    cout << _output_file_name.c_str() << endl;
    
    ofstream output_file;
    output_file.open(_output_file_name.c_str());
    for(int i = 0; i < vertices_count; i++)
    {
        output_file << connections_data[i] << " ";
    }
    output_file.close();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
