//
//  gapbs_export.h
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 08/06/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef gapbs_export_h
#define gapbs_export_h

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void export_to_gapbs_text_unweighted(EdgesListGraph<_TVertexValue, _TEdgeWeight> &_graph, string _gapbs_file_name)
{
    int vertices_count = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count();
    int *src_ids = _graph.get_src_ids();
    int *dst_ids = _graph.get_dst_ids();
    
    ofstream gapbs_file(_gapbs_file_name.c_str());
    
    for(long long i = 0; i < edges_count; i++)
    {
        int src_id = src_ids[i];
        int dst_id = dst_ids[i];
        gapbs_file << src_id << " " << dst_id << '\n';
    }
    
    gapbs_file.close();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#endif /* gapbs_export_h */
