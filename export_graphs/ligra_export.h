//
//  ligra_export.h
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 05/06/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef ligra_export_h
#define ligra_export_h

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void export_to_ligra_text_unweighted(EdgesListGraph<_TVertexValue, _TEdgeWeight> &_graph, string _ligra_file_name)
{
    int vertices_count = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count();
    int *src_ids = _graph.get_src_ids();
    int *dst_ids = _graph.get_dst_ids();
    
    vector<vector<int> > adj_graph(vertices_count);
    for(long long i = 0; i < edges_count; i++)
    {
        int src_id = src_ids[i];
        int dst_id = dst_ids[i];
        adj_graph[src_id].push_back(dst_id);
    }
    
    int cur_offset = 0;
    vector<int> offsets(vertices_count);
    for(int i = 0; i < vertices_count; i++)
    {
        offsets[i] = cur_offset;
        cur_offset += adj_graph[i].size();
    }
    
    ofstream ligra_file(_ligra_file_name.c_str());
    
    ligra_file << "AdjacencyGraph" << '\n';
    ligra_file << vertices_count << '\n';
    ligra_file << edges_count << '\n';
    
    for(int i = 0; i < vertices_count; i++)
    {
        ligra_file << offsets[i] << '\n';
    }
    
    for(int i = 0; i < vertices_count; i++)
    {
        for(int j = 0; j < adj_graph[i].size(); j++)
        {
            ligra_file << adj_graph[i][j] << '\n';
        }
    }
    
    ligra_file.close();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* ligra_export_h */
