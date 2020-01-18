//
//  vectorised_CSR_graph.hpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 19/04/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef vectorised_CSR_graph_hpp
#define vectorised_CSR_graph_hpp

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
VectorisedCSRGraph<_TVertexValue, _TEdgeWeight>::VectorisedCSRGraph(int _vertices_count, long long _edges_count)
{
    this->graph_type = GraphTypeVectorisedCSR;
    
    supported_vector_length = 1;
    vertices_state          = VERTICES_UNSORTED;
    edges_state             = EDGES_UNSORTED;
    
    threads_count = 1;
    
    this->alloc(_vertices_count, _edges_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
VectorisedCSRGraph<_TVertexValue, _TEdgeWeight>::~VectorisedCSRGraph()
{
    this->free();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void VectorisedCSRGraph<_TVertexValue, _TEdgeWeight>::alloc(int _vertices_count, long long _edges_count,
                                                            int _number_of_vertices_in_first_part)
{
    this->vertices_count    = _vertices_count;
    this->edges_count       = _edges_count;
    
    number_of_vertices_in_first_part      = _number_of_vertices_in_first_part;
    vertices_in_vector_segments = this->vertices_count - _number_of_vertices_in_first_part;
    
    if(vertices_in_vector_segments % this->supported_vector_length != 0)
        throw "ERROR: vertices_count or fisrt_part_border values are incorrect for VectorisedCSRGraph";
    
    this->vertex_values  = new _TVertexValue[this->vertices_count];
    reordered_vertex_ids = new int[this->vertices_count];
    
    vector_segments_count = vertices_in_vector_segments / this->supported_vector_length;
    
    first_part_ptrs = new long long[number_of_vertices_in_first_part+1];
    first_part_sizes = new int[number_of_vertices_in_first_part];
    
    vector_group_ptrs  = new long long[vector_segments_count];
    vector_group_sizes = new int[vector_segments_count];
    
    outgoing_ids        = new int[this->edges_count];
    outgoing_weights    = new _TEdgeWeight[this->edges_count];
    
    incoming_sizes_per_vertex = new int[this->vertices_count];
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void VectorisedCSRGraph<_TVertexValue, _TEdgeWeight>::free()
{
    delete []this->vertex_values;
    
    delete []reordered_vertex_ids;
    
    delete []vector_group_ptrs;
    delete []vector_group_sizes;
    
    delete []first_part_ptrs;
    delete []first_part_sizes;
    
    delete []outgoing_ids;
    delete []outgoing_weights;
    
    delete []incoming_sizes_per_vertex;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void VectorisedCSRGraph<_TVertexValue, _TEdgeWeight>::resize(int _vertices_count, long long _edges_count,
                                                             int _number_of_vertices_in_first_part)
{
    this->free();
    this->alloc(_vertices_count, _edges_count, _number_of_vertices_in_first_part);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void VectorisedCSRGraph<_TVertexValue, _TEdgeWeight>::print()
{
    cout << "VectorisedCSRGraph format" << endl;
    
    cout << "VerticesState: " << this->vertices_state << endl;
    cout << "EdgesState: " << this->edges_state << endl;
    cout << "SupportedVectorLength: " << this->supported_vector_length << endl << endl;
    
    cout << "Vertices data: " << endl;
    for(int i = 0; i < this->vertices_count; i++)
        cout << this->vertex_values[i] << " ";
    cout << endl;
    
    cout << "Edges data: " << endl;
    for(int cur_vertex = 0; cur_vertex < number_of_vertices_in_first_part; cur_vertex++)
    {
        cout << "vertex " << cur_vertex << " connected to: ";
        long long int start = first_part_ptrs[cur_vertex];
        for(int edge_pos = 0; edge_pos < first_part_sizes[cur_vertex]; edge_pos++)
        {
            cout << "(" << outgoing_ids[start + edge_pos] << "," << outgoing_weights[start + edge_pos] << ")" << " ";
        }
        cout << endl;
    }
    
    cout << " ------------ vector segment part ------------ " << endl;
    
    for(int cur_vector_segment = 0; cur_vector_segment < vector_segments_count; cur_vector_segment++)
    {
        int vec_start = cur_vector_segment * supported_vector_length + number_of_vertices_in_first_part;
        long long edge_start = vector_group_ptrs[cur_vector_segment];
        int cur_max_connections_count = vector_group_sizes[cur_vector_segment];
        
        for(int i = 0; i < supported_vector_length; i++)
        {
            int cur_vertex = vec_start + i;
            cout << "vertex " << cur_vertex << " connected to: ";
            
            for(int edge_pos = 0; edge_pos < cur_max_connections_count; edge_pos++)
            {
                cout << "(" << outgoing_ids[edge_start + edge_pos * supported_vector_length + i] << "," << outgoing_weights[edge_start + edge_pos * supported_vector_length + i] << ")" << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void VectorisedCSRGraph<_TVertexValue, _TEdgeWeight>::save_to_graphviz_file(string _file_name, VisualisationMode _visualisation_mode)
{
    ofstream dot_output(_file_name.c_str());
    
    string connection;
    dot_output << "digraph G {" << endl;
    connection = " -> ";
    
    for(int i = 0; i < this->vertices_count; i++)
    {
        dot_output << i << " [label= \"id=" << i << ", value=" << this->vertex_values[i] << "\"] "<< endl;
    }
    
    for (int src_id = 0; src_id < this->vertices_count; src_id++)
    {
        long long edge_pos = get_vertex_pointer(src_id);
        int connections_count = get_vector_connections_count(src_id);
        
        int multiplier = 1;
        if(src_id >= number_of_vertices_in_first_part)
            multiplier = this->supported_vector_length;
        
        for(int i = 0; i < connections_count; i++)
        {
            int dst_id = outgoing_ids[edge_pos + i * multiplier];
            _TEdgeWeight weight = outgoing_weights[edge_pos + i * multiplier];
            
            dot_output << src_id << connection << dst_id << " [label = \" " << weight << " \"];" << endl;
        }
    }
    
    dot_output << "}";
    dot_output.close();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _TVertexValue, typename _TEdgeWeight>
void VectorisedCSRGraph<_TVertexValue, _TEdgeWeight>::estimate_gpu_thresholds()
{
    gpu_grid_threshold_vertex = 0;
    gpu_block_threshold_vertex = 0;
    gpu_warp_threshold_vertex = 0;
    gpu_warp_threshold_vertex = 0;
    
    for(int i = 0; i < (number_of_vertices_in_first_part - 1); i++)
    {
        if((first_part_sizes[i] > GPU_GRID_THREASHOLD_VALUE) && (first_part_sizes[i+1] <= GPU_GRID_THREASHOLD_VALUE))
        {
            gpu_grid_threshold_vertex = i + 1;
        }
        if((first_part_sizes[i] > GPU_BLOCK_THREASHOLD_VALUE) && (first_part_sizes[i+1] <= GPU_BLOCK_THREASHOLD_VALUE))
        {
            gpu_block_threshold_vertex = i + 1;
        }
        if((first_part_sizes[i] > GPU_WARP_THREASHOLD_VALUE) && (first_part_sizes[i+1] <= GPU_WARP_THREASHOLD_VALUE))
        {
            gpu_warp_threshold_vertex = i + 1;
        }
    }
    gpu_warp_threshold_vertex = number_of_vertices_in_first_part;
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
bool VectorisedCSRGraph<_TVertexValue, _TEdgeWeight>::save_to_binary_file(string _file_name)
{
    FILE * graph_file = fopen(_file_name.c_str(), "wb");
    if(graph_file == NULL)
        return false;
    
    int vertices_count = this->vertices_count;
    long long edges_count = this->edges_count;
    fwrite(reinterpret_cast<const void*>(&vertices_count), sizeof(int), 1, graph_file);
    fwrite(reinterpret_cast<const void*>(&edges_count), sizeof(long long), 1, graph_file);
    fwrite(reinterpret_cast<const void*>(&number_of_vertices_in_first_part), sizeof(int), 1, graph_file);
    
    fwrite(reinterpret_cast<const void*>(&(this->graph_type)), sizeof(GraphType), 1, graph_file);
    fwrite(reinterpret_cast<const void*>(&vertices_state), sizeof(VerticesState), 1, graph_file);
    fwrite(reinterpret_cast<const void*>(&edges_state), sizeof(EdgesState), 1, graph_file);
    
    fwrite(reinterpret_cast<const void*>(&supported_vector_length), sizeof(int), 1, graph_file);
    fwrite(reinterpret_cast<const void*>(&vector_segments_count), sizeof(int), 1, graph_file);
    
    fwrite(reinterpret_cast<const void*>(this->vertex_values), sizeof(_TVertexValue), vertices_count, graph_file);
    fwrite(reinterpret_cast<const void*>(reordered_vertex_ids), sizeof(int), vertices_count, graph_file);
    
    fwrite(reinterpret_cast<const void*>(first_part_ptrs), sizeof(long long), number_of_vertices_in_first_part + 1, graph_file);
    fwrite(reinterpret_cast<const void*>(first_part_sizes), sizeof(int), number_of_vertices_in_first_part, graph_file);
    
    fwrite(reinterpret_cast<const void*>(vector_group_ptrs), sizeof(long long), vector_segments_count, graph_file);
    fwrite(reinterpret_cast<const void*>(vector_group_sizes), sizeof(int), vector_segments_count, graph_file);
    
    fwrite(reinterpret_cast<const void*>(outgoing_ids), sizeof(int), edges_count, graph_file);
    fwrite(reinterpret_cast<const void*>(outgoing_weights), sizeof(_TEdgeWeight), edges_count, graph_file);
    
    fwrite(reinterpret_cast<const void*>(incoming_sizes_per_vertex), sizeof(int), this->vertices_count, graph_file);
    
    fclose(graph_file);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
bool VectorisedCSRGraph<_TVertexValue, _TEdgeWeight>::load_from_binary_file(string _file_name)
{
    FILE * graph_file = fopen(_file_name.c_str(), "rb");
    if(graph_file == NULL)
        return false;
    
    fread(reinterpret_cast<void*>(&this->vertices_count), sizeof(int), 1, graph_file);
    fread(reinterpret_cast<void*>(&this->edges_count), sizeof(long long), 1, graph_file);
    fread(reinterpret_cast<void*>(&number_of_vertices_in_first_part), sizeof(int), 1, graph_file);
    
    this->resize(this->vertices_count, this->edges_count, number_of_vertices_in_first_part);
    
    fread(reinterpret_cast<void*>(&(this->graph_type)), sizeof(GraphType), 1, graph_file);
    fread(reinterpret_cast<void*>(&vertices_state), sizeof(VerticesState), 1, graph_file);
    fread(reinterpret_cast<void*>(&edges_state), sizeof(EdgesState), 1, graph_file);
    
    if(this->graph_type != GraphTypeVectorisedCSR)
        throw "ERROR: loaded incorrect type of graph into VectorisedCSRGraph container";
    
    fread(reinterpret_cast<void*>(&supported_vector_length), sizeof(int), 1, graph_file);
    fread(reinterpret_cast<void*>(&vector_segments_count), sizeof(int), 1, graph_file);
    
    fread(reinterpret_cast<void*>(this->vertex_values), sizeof(_TVertexValue), this->vertices_count, graph_file);
    fread(reinterpret_cast<void*>(reordered_vertex_ids), sizeof(int), this->vertices_count, graph_file);
    
    fread(reinterpret_cast<void*>(first_part_ptrs), sizeof(long long), number_of_vertices_in_first_part + 1, graph_file);
    fread(reinterpret_cast<void*>(first_part_sizes), sizeof(int), number_of_vertices_in_first_part, graph_file);
    
    fread(reinterpret_cast<void*>(vector_group_ptrs), sizeof(long long), vector_segments_count, graph_file);
    fread(reinterpret_cast<void*>(vector_group_sizes), sizeof(int), vector_segments_count, graph_file);
    
    fread(reinterpret_cast<void*>(outgoing_ids), sizeof(int), this->edges_count, graph_file);
    fread(reinterpret_cast<void*>(outgoing_weights), sizeof(_TEdgeWeight), this->edges_count, graph_file);
    
    fread(reinterpret_cast<void*>(incoming_sizes_per_vertex), sizeof(int), this->vertices_count, graph_file);
    
    #ifdef __USE_GPU__
    estimate_gpu_thresholds();
    #endif
    
    fclose(graph_file);
    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _TVertexValue, typename _TEdgeWeight>
void VectorisedCSRGraph<_TVertexValue, _TEdgeWeight>::move_to_device()
{
    if(this->graph_on_device)
    {
        return;
    }
    
    this->graph_on_device = true;
    
    move_array_to_device<_TVertexValue>(&(this->vertex_values), this->vertices_count);
    move_array_to_device<int>(&reordered_vertex_ids, this->vertices_count);
    
    move_array_to_device<long long>(&first_part_ptrs, number_of_vertices_in_first_part + 1);
    move_array_to_device<int>(&first_part_sizes, number_of_vertices_in_first_part);
    
    move_array_to_device<long long>(&vector_group_ptrs, vector_segments_count);
    move_array_to_device<int>(&vector_group_sizes, vector_segments_count);
    
    move_array_to_device<int>(&outgoing_ids, this->edges_count);
    move_array_to_device<_TEdgeWeight>(&outgoing_weights, this->edges_count);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _TVertexValue, typename _TEdgeWeight>
void VectorisedCSRGraph<_TVertexValue, _TEdgeWeight>::move_to_host()
{
    if(!this->graph_on_device)
    {
        return;
    }
    
    this->graph_on_device = false;
    
    move_array_to_host<_TVertexValue>(&(this->vertex_values), this->vertices_count);
    move_array_to_host<int>(&reordered_vertex_ids, this->vertices_count);
    
    move_array_to_host<long long>(&first_part_ptrs, number_of_vertices_in_first_part + 1);
    move_array_to_host<int>(&first_part_sizes, number_of_vertices_in_first_part);

    move_array_to_host<long long>(&vector_group_ptrs, vector_segments_count);
    move_array_to_host<int>(&vector_group_sizes, vector_segments_count);
    
    move_array_to_host<int>(&outgoing_ids, this->edges_count);
    move_array_to_host<_TEdgeWeight>(&outgoing_weights, this->edges_count);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* vectorised_CSR_graph_hpp */
