//
//  vectorised_CSR_graph.hpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 14/04/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef extended_CSR_graph_hpp
#define extended_CSR_graph_hpp

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::ExtendedCSRGraph(int _vertices_count, long long _edges_count)
{
    this->graph_type = GraphTypeExtendedCSR;
    
    supported_vector_length = 1;
    vertices_state = VERTICES_UNSORTED;
    edges_state = EDGES_UNSORTED;
    
    alloc(_vertices_count, _edges_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::~ExtendedCSRGraph()
{
    free();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::alloc(int _vertices_count, long long _edges_count)
{
    this->vertices_count = _vertices_count;
    this->edges_count = _edges_count;
    
    reordered_vertex_ids = new int[this->vertices_count];
    outgoing_ptrs        = new long long[this->vertices_count + 1];
    outgoing_ids         = new int[this->edges_count];
    outgoing_weights     = new _TEdgeWeight[this->edges_count];
    incoming_degrees     = new int[this->vertices_count];
    this->vertex_values  = new _TVertexValue[this->vertices_count];
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::free()
{
    delete []reordered_vertex_ids;
    delete []outgoing_ptrs;
    delete []outgoing_ids;
    delete []outgoing_weights;
    delete []incoming_degrees;
    delete []this->vertex_values;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::resize(int _vertices_count, long long _edges_count)
{
    this->free();
    this->alloc(_vertices_count, _edges_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::import_graph(EdgesListGraph<_TVertexValue, _TEdgeWeight>
                                                                 &_old_graph,
                                                                 VerticesState _vertices_state,
                                                                 EdgesState _edges_state,
                                                                 int _supported_vector_length,
                                                                 SupportedTraversalType _traversal_type)
{
    // set optimisation parameters
    this->vertices_state          = _vertices_state;
    this->edges_state             = _edges_state;
    this->supported_vector_length = _supported_vector_length;
    
    // create tmp graph
    int tmp_vertices_count = _old_graph.get_vertices_count();
    long long tmp_edges_count = _old_graph.get_edges_count();
    
    vector<vector<TempEdgeData<_TEdgeWeight> > >tmp_graph(tmp_vertices_count);
    
    _TVertexValue *old_vertex_values = _old_graph.get_vertex_values();
    int *old_src_ids = _old_graph.get_src_ids();
    int *old_dst_ids = _old_graph.get_dst_ids();
    _TEdgeWeight *old_weights = _old_graph.get_weights();
    
    for(long long int i = 0; i < tmp_edges_count; i++)
    {
        int src_id = old_src_ids[i];
        int dst_id = old_dst_ids[i];
        _TEdgeWeight weight = old_weights[i];
        
        if(_traversal_type == PUSH_TRAVERSAL)
            tmp_graph[src_id].push_back(TempEdgeData<_TEdgeWeight>(dst_id, weight));
        else if(_traversal_type == PULL_TRAVERSAL)
            tmp_graph[dst_id].push_back(TempEdgeData<_TEdgeWeight>(src_id, weight));
    }
    
    // sort all vertices now
    vector<pair<int, int> > pairs(tmp_vertices_count);
    for(int i = 0; i < tmp_vertices_count; ++i)
        pairs[i] = make_pair(tmp_graph[i].size(), i);
    
    if(vertices_state == VERTICES_SORTED)
    {
        sort(pairs.begin(), pairs.end());
        reverse(pairs.begin(), pairs.end());
    }
    
    // save old indexes array
    int *old_indexes = new int[tmp_vertices_count];
    for(int i = 0; i < tmp_vertices_count; i++)
    {
        old_indexes[i] = pairs[i].second;
    }
    
    // need to reoerder all data arrays in 2 steps
    vector<vector<TempEdgeData<_TEdgeWeight> > > new_tmp_graph(tmp_vertices_count);
    for(int i = 0; i < tmp_vertices_count; i++)
    {
        new_tmp_graph[i] = tmp_graph[old_indexes[i]];
    }
    
    for(int i = 0; i < tmp_vertices_count; i++)
    {
        tmp_graph[i] = new_tmp_graph[i];
    }
    
    // get correct reordered array
    int *tmp_reordered_vertex_ids = new int[tmp_vertices_count];
    for(int i = 0; i < tmp_vertices_count; i++)
    {
        tmp_reordered_vertex_ids[old_indexes[i]] = i;
    }
    
    delete []old_indexes;
    
    // sort adjacent ids locally for each vertex
    for(int cur_vertex = 0; cur_vertex < tmp_vertices_count; cur_vertex++)
    {
        int src_id = cur_vertex;
        for(int i = 0; i < tmp_graph[src_id].size(); i++)
        {
            tmp_graph[src_id][i].dst_id = tmp_reordered_vertex_ids[tmp_graph[src_id][i].dst_id];
        }
        if(edges_state == EDGES_SORTED)
        {
            std::sort(tmp_graph[src_id].begin(), tmp_graph[src_id].end(), edge_cmp<_TEdgeWeight>);
        }
        else if(edges_state == EDGES_RANDOM_SHUFFLED)
        {
            std::random_shuffle(tmp_graph[src_id].begin(), tmp_graph[src_id].end());
        }
    }
    
    // vector flatten here
    for(int i = 0; i < tmp_vertices_count; i += supported_vector_length)
    {
        int max_segment_size = tmp_graph[i].size();
        for(int j = 0; j < supported_vector_length; j++)
        {
            while(tmp_graph[i + j].size() < max_segment_size)
            {
                tmp_graph[i + j].push_back(TempEdgeData<_TEdgeWeight>(i + j, 0.0));
                tmp_edges_count++;
            }
        }
    }
    
    // get new pointers
    this->resize(tmp_vertices_count, tmp_edges_count);
    
    // save optimised graph
    long long current_edge = 0;
    this->outgoing_ptrs[0] = current_edge;
    for(int cur_vertex = 0; cur_vertex < this->vertices_count; cur_vertex++)
    {
        int src_id = cur_vertex;
        this->vertex_values[cur_vertex] = old_vertex_values[cur_vertex];
        this->reordered_vertex_ids[cur_vertex] = tmp_reordered_vertex_ids[cur_vertex];
        
        for(int i = 0; i < tmp_graph[src_id].size(); i++)
        {
            this->outgoing_ids[current_edge] = tmp_graph[src_id][i].dst_id;
            this->outgoing_weights[current_edge] = tmp_graph[src_id][i].weight;
            current_edge++;
        }
        this->outgoing_ptrs[cur_vertex + 1] = current_edge;
    }
    
    delete []tmp_reordered_vertex_ids;
    
    calculate_incoming_degrees();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::print()
{
    cout << "ExtendedCSRGraph format" << endl;
    
    cout << "VerticesState: " << this->vertices_state << endl;
    cout << "EdgesState: " << this->edges_state << endl;
    cout << "SupportedVectorLength: " << this->supported_vector_length << endl << endl;
    
    cout << "Vertices data: " << endl;
    for(int i = 0; i < this->vertices_count; i++)
        cout << this->vertex_values[i] << " ";
    cout << endl;

    cout << "Edges data: " << endl;
    for(int cur_vertex = 0; cur_vertex < this->vertices_count; cur_vertex++)
    {
        cout << "vertex " << cur_vertex << " connected to: ";
        for(long long edge_pos = outgoing_ptrs[cur_vertex]; edge_pos < outgoing_ptrs[cur_vertex + 1]; edge_pos++)
            cout << "(" << outgoing_ids[edge_pos] << "," << outgoing_weights[edge_pos] << ")" << " ";
        cout << endl;
    }
    cout << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::save_to_graphviz_file(string _file_name, VisualisationMode _visualisation_mode)
{
    ofstream dot_output(_file_name.c_str());
    
    string connection;
    if(_visualisation_mode == VISUALISE_AS_DIRECTED)
    {
        dot_output << "digraph G {" << endl;
        connection = " -> ";
    }
    else if(_visualisation_mode == VISUALISE_AS_UNDIRECTED)
    {
        dot_output << "graph G {" << endl;
        connection = " -- ";
    }
    
        
    for(int i = 0; i < this->vertices_count; i++)
    {
        dot_output << i << " [label= \"id=" << i << ", value=" << this->vertex_values[i] << "\"] "<< endl;
        //dot_output << i << " [label=" << this->vertex_values[i] << "]"<< endl;
    }
    
    for(int cur_vertex = 0; cur_vertex < this->vertices_count; cur_vertex++)
    {
        int src_id = cur_vertex;
        for(long long edge_pos = outgoing_ptrs[cur_vertex]; edge_pos < outgoing_ptrs[cur_vertex + 1]; edge_pos++)
        {
            int dst_id = outgoing_ids[edge_pos];
            _TEdgeWeight weight = outgoing_weights[edge_pos];
            if(src_id != dst_id)
            {
                if(_visualisation_mode == VISUALISE_AS_UNDIRECTED)
                {
                    if(src_id < dst_id)
                    {
                        dot_output << src_id << connection << dst_id << " [label = \" " << weight << " \"];" << endl;
                    }
                }
                else
                {
                    dot_output << src_id << connection << dst_id << " [label = \" " << weight << " \"];" << endl;
                }
            }
        }
    }
    
    dot_output << "}";
    dot_output.close();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
bool ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::save_to_binary_file(string _file_name)
{
    FILE * graph_file = fopen(_file_name.c_str(), "wb");
    if(graph_file == NULL)
        return false;
    
    int vertices_count = this->vertices_count;
    long long edges_count = this->edges_count;
    fwrite(reinterpret_cast<const void*>(&vertices_count), sizeof(int), 1, graph_file);
    fwrite(reinterpret_cast<const void*>(&edges_count), sizeof(long long), 1, graph_file);
    
    fwrite(reinterpret_cast<const void*>(&(this->graph_type)), sizeof(GraphType), 1, graph_file);
    fwrite(reinterpret_cast<const void*>(&vertices_state), sizeof(VerticesState), 1, graph_file);
    fwrite(reinterpret_cast<const void*>(&edges_state), sizeof(EdgesState), 1, graph_file);
    fwrite(reinterpret_cast<const void*>(&supported_vector_length), sizeof(int), 1, graph_file);
    
    fwrite(reinterpret_cast<const char*>(this->vertex_values), sizeof(_TVertexValue), vertices_count, graph_file);
    fwrite(reinterpret_cast<const char*>(reordered_vertex_ids), sizeof(int), vertices_count, graph_file);
    fwrite(reinterpret_cast<const char*>(outgoing_ptrs), sizeof(long long), vertices_count + 1, graph_file);
    
    fwrite(reinterpret_cast<const char*>(outgoing_ids), sizeof(int), edges_count, graph_file);
    fwrite(reinterpret_cast<const char*>(outgoing_weights), sizeof(_TEdgeWeight), edges_count, graph_file);
    
    fclose(graph_file);
    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
bool ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::load_from_binary_file(string _file_name)
{
    FILE * graph_file = fopen(_file_name.c_str(), "rb");
    if(graph_file == NULL)
        return false;
    
    fread(reinterpret_cast<void*>(&this->vertices_count), sizeof(int), 1, graph_file);
    fread(reinterpret_cast<void*>(&this->edges_count), sizeof(long long), 1, graph_file);
    
    this->resize(this->vertices_count, this->edges_count);
    
    fread(reinterpret_cast<void*>(&(this->graph_type)), sizeof(GraphType), 1, graph_file);
    
    if(this->graph_type != GraphTypeExtendedCSR)
        throw "ERROR: loaded incorrect type of graph into ExtendedCSRGraph container";
    
    fread(reinterpret_cast<void*>(&vertices_state), sizeof(VerticesState), 1, graph_file);
    fread(reinterpret_cast<void*>(&edges_state), sizeof(EdgesState), 1, graph_file);
    fread(reinterpret_cast<void*>(&supported_vector_length), sizeof(int), 1, graph_file);
    
    fread(reinterpret_cast<char*>(this->vertex_values), sizeof(_TVertexValue), this->vertices_count, graph_file);
    fread(reinterpret_cast<char*>(reordered_vertex_ids), sizeof(int), this->vertices_count, graph_file);
    fread(reinterpret_cast<char*>(outgoing_ptrs), sizeof(long long), (this->vertices_count + 1), graph_file);
    
    fread(reinterpret_cast<char*>(outgoing_ids), sizeof(int), this->edges_count, graph_file);
    fread(reinterpret_cast<char*>(outgoing_weights), sizeof(_TEdgeWeight), this->edges_count, graph_file);
    
    fclose(graph_file);
    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::calculate_incoming_degrees()
{
    int vertices_count = this->vertices_count;
    for(int i = 0; i < vertices_count; i++)
    {
        incoming_degrees[i] = 0;
    }
    
    for(int src_id = 0; src_id < vertices_count; src_id++)
    {
        long long edge_start = outgoing_ptrs[src_id];
        int connections_count = outgoing_ptrs[src_id + 1] - outgoing_ptrs[src_id];
        for(int edge_pos = 0; edge_pos < connections_count; edge_pos++)
        {
            int dst_id = outgoing_ids[edge_start + edge_pos];
            incoming_degrees[dst_id]++;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* extended_CSR_graph_hpp */
