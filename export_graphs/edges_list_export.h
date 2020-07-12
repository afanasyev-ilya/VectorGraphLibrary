#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void export_to_edges_list_unweighted(EdgesListGraph<_TVertexValue, _TEdgeWeight> &_graph, string _gapbs_file_name, bool _use_mtx_header)
{
    int vertices_count = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count();
    int *src_ids = _graph.get_src_ids();
    int *dst_ids = _graph.get_dst_ids();

    bool vertex_starts_from_zero = false;

    for(long long i = 0; i < edges_count; i++)
    {
        int src_id = src_ids[i];
        int dst_id = dst_ids[i];
        if(src_id == 0)
            vertex_starts_from_zero = true;
        if(dst_id == 0)
            vertex_starts_from_zero = true;
    }

    int shift = 0;
    if(vertex_starts_from_zero)
        shift = 1;
    
    ofstream gapbs_file(_gapbs_file_name.c_str());
    
    if(_use_mtx_header)
        gapbs_file << vertices_count + shift << " " << vertices_count + shift << " " << edges_count << '\n';
    
    for(long long i = 0; i < edges_count; i++)
    {
        int src_id = src_ids[i];
        int dst_id = dst_ids[i];
        gapbs_file << src_id + shift << " " << dst_id + shift << '\n';
    }
    
    gapbs_file.close();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void export_to_edges_list_weighted(EdgesListGraph<_TVertexValue, _TEdgeWeight> &_graph, string _gapbs_file_name, bool _use_mtx_header)
{
    int vertices_count = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count();
    int *src_ids = _graph.get_src_ids();
    int *dst_ids = _graph.get_dst_ids();
    _TEdgeWeight *weights = _graph.get_weights();

    bool vertex_starts_from_zero = false;

    for(long long i = 0; i < edges_count; i++)
    {
        int src_id = src_ids[i];
        int dst_id = dst_ids[i];
        if(src_id == 0)
            vertex_starts_from_zero = true;
        if(dst_id == 0)
            vertex_starts_from_zero = true;
    }

    int shift = 0;
    if(vertex_starts_from_zero)
        shift = 1;

    ofstream gapbs_file(_gapbs_file_name.c_str());

    if(_use_mtx_header)
        gapbs_file << vertices_count + shift << " " << vertices_count + shift << " " << edges_count << '\n';

    for(long long i = 0; i < edges_count; i++)
    {
        int src_id = src_ids[i];
        int dst_id = dst_ids[i];

        _TEdgeWeight weight = weights[i];
        if(weight < 0 || weight > MAX_WEIGHT)
            weight = rand()%MAX_WEIGHT;
        gapbs_file << src_id + shift << " " << dst_id + shift << " " << weight << '\n';
    }

    gapbs_file.close();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

