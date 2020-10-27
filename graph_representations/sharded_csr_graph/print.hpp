#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ShardedCSRGraph::print()
{

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void ShardedCSRGraph::print_in_csr_format(EdgesArray_Sharded<_T> &_weights)
{
    if(can_use_scatter())
    {
        for(int sh = 0; sh < shards_number; sh++)
        {
            UndirectedCSRGraph *shard_graph = &(outgoing_shards[sh]);
            for(int v = 0; v < shard_graph->get_vertices_count(); v++)
            {
                int start = shard_graph->get_vertex_pointers()[v];
                int end = shard_graph->get_vertex_pointers()[v + 1];
                int connections = end - start;
                cout << "vertex " << shard_graph->reorder_to_original(v) << " is connected to: ";
                for(int edge_pos = start; edge_pos < end; edge_pos++)
                {
                    int dst_id = shard_graph->get_adjacent_ids()[edge_pos];
                    int weight_pos = this->get_shard_shift(sh) + edge_pos;
                    cout << "(" << shard_graph->reorder_to_original(dst_id) << ", " << _weights[weight_pos] << ") ";
                }
                cout << endl;
            }
        }
        cout << endl;
    }
    // TODO incoming part
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
