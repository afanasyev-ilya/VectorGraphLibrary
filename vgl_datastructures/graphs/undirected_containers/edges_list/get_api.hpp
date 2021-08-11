#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

any_arch_func int EdgesListGraph::get_edge_dst(int _src_id, int _edge_pos)
{
    int cnt = 0;
    for(long long i = 0; i < this->edges_count; i++)
    {
        int src_id = src_ids[i];
        if(src_id == _src_id)
        {
            if(cnt == _edge_pos)
                return dst_ids[i];
            cnt++;
        }
    }
    #ifndef __USE_GPU__
    throw "Error in EdgesListGraph::get_edge_dst : _edge_pos out of range";
    #endif
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
