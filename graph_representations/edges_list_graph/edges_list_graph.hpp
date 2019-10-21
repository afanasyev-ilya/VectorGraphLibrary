//
//  edges_list_graph.hpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 14/04/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef edges_list_graph_hpp
#define edges_list_graph_hpp

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
EdgesListGraph<_TVertexValue, _TEdgeWeight>::EdgesListGraph(int _vertices_count = 1, long long _edges_count = 1)
{
    this->graph_type = GraphTypeEdgesList;
    
    alloc(_vertices_count, _edges_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
EdgesListGraph<_TVertexValue, _TEdgeWeight>::~EdgesListGraph()
{
    free();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void EdgesListGraph<_TVertexValue, _TEdgeWeight>::alloc(int _vertices_count, long long _edges_count)
{
    this->vertices_count = _vertices_count;
    this->edges_count    = _edges_count;
    this->vertex_values  = new _TVertexValue[this->vertices_count];
    src_ids              = new int[this->edges_count];
    dst_ids              = new int[this->edges_count];
    weights              = new _TEdgeWeight[this->edges_count];
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void EdgesListGraph<_TVertexValue, _TEdgeWeight>::free()
{
    if(this->vertex_values != NULL)
        delete []this->vertex_values;
    if(src_ids != NULL)
        delete []src_ids;
    if(dst_ids != NULL)
        delete []dst_ids;
    if(weights != NULL)
        delete []weights;
    
    this->vertex_values = NULL;
    src_ids = NULL;
    dst_ids = NULL;
    weights = NULL;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void EdgesListGraph<_TVertexValue, _TEdgeWeight>::resize(int _vertices_count, long long _edges_count)
{
    this->free();
    this->alloc(_vertices_count, _edges_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void EdgesListGraph<_TVertexValue, _TEdgeWeight>::print()
{
    cout << "Graph in edges list format" << endl;
    cout << "Vertices data: " << endl;
    for(int i = 0; i < this->vertices_count; i++)
        cout << this->vertex_values[i] << " ";
    cout << endl;
    cout << "Edges data: " << endl;
    for(long long int i = 0; i < this->edges_count; i++)
        cout << src_ids[i] << " " << dst_ids[i] << " " << weights[i] << endl;
    cout << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void EdgesListGraph<_TVertexValue, _TEdgeWeight>::save_to_graphviz_file(string _file_name, VisualisationMode _visualisation_mode)
{
    ofstream dot_output(_file_name.c_str());
    
    string connection;
    dot_output << "digraph G {" << endl;
    connection = " -> ";
    
    for(int i = 0; i < this->vertices_count; i++)
    {
        dot_output << i << " [label= \"id=" << i << ", value=" << this->vertex_values[i] << "\"] "<< endl;
        //dot_output << i << " [label=" << this->vertex_values[i] << "]"<< endl;
    }
    
    for(long long i = 0; i < this->edges_count; i++)
    {
        int src_id = src_ids[i];
        int dst_id = dst_ids[i];
        _TEdgeWeight weight = weights[i];
        dot_output << src_id << connection << dst_id << " [label = \" " << weight << " \"];" << endl;
    }
    
    dot_output << "}";
    dot_output.close();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
bool EdgesListGraph<_TVertexValue, _TEdgeWeight>::save_to_binary_file(string _file_name)
{
    FILE * graph_file = fopen(_file_name.c_str(), "wb");
    if(graph_file == NULL)
        return false;
    
    int vertices_count = this->vertices_count;
    long long edges_count = this->edges_count;
    fwrite(reinterpret_cast<const void*>(&vertices_count), sizeof(int), 1, graph_file);
    fwrite(reinterpret_cast<const void*>(&edges_count), sizeof(long long), 1, graph_file);

    fwrite(reinterpret_cast<const void*>(src_ids), sizeof(int), this->edges_count, graph_file);
    fwrite(reinterpret_cast<const void*>(dst_ids), sizeof(int), this->edges_count, graph_file);
    fwrite(reinterpret_cast<const void*>(weights), sizeof(_TEdgeWeight), this->edges_count, graph_file);

    fclose(graph_file);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
bool EdgesListGraph<_TVertexValue, _TEdgeWeight>::load_from_binary_file(string _file_name)
{
    FILE * graph_file = fopen(_file_name.c_str(), "rb");
    if(graph_file == NULL)
        return false;
    
    fread(reinterpret_cast<void*>(&this->vertices_count), sizeof(int), 1, graph_file);
    fread(reinterpret_cast<void*>(&this->edges_count), sizeof(long long), 1, graph_file);
    
    fread(reinterpret_cast<void*>(src_ids), sizeof(int), this->edges_count, graph_file);
    fread(reinterpret_cast<void*>(dst_ids), sizeof(int), this->edges_count, graph_file);
    fread(reinterpret_cast<void*>(weights), sizeof(_TEdgeWeight), this->edges_count, graph_file);
    
    fclose(graph_file);
    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* edges_list_graph_hpp */
