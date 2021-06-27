#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ShardedCSRGraph::print()
{
    if(can_use_scatter())
    {
        for (int sh = 0; sh < shards_number; sh++)
        {
            outgoing_shards[sh].print();
        }
    }
    if(can_use_gather())
    {
        for (int sh = 0; sh < shards_number; sh++)
        {
            incoming_shards[sh].print();
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void ShardedCSRGraph::print_in_csr_format(EdgesArray_Sharded<_T> &_weights)
{
    if(can_use_scatter())
    {
        for(int sh = 0; sh < shards_number; sh++)
        {
            UndirectedVectCSRGraph *shard_graph = &(outgoing_shards[sh]);
            for(int v = 0; v < shard_graph->get_vertices_count(); v++)
            {
                int start = shard_graph->get_vertex_pointers()[v];
                int end = shard_graph->get_vertex_pointers()[v + 1];
                int connections = end - start;
                cout << "vertex " << shard_graph->reorder_to_original(v) << " is connected to: ";
                for(int edge_pos = start; edge_pos < end; edge_pos++)
                {
                    int dst_id = shard_graph->get_adjacent_ids()[edge_pos];
                    int weight_pos = this->get_shard_shift(sh, SCATTER) + edge_pos; // TODO SCATTER FIX
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

size_t ShardedCSRGraph::get_size()
{
    size_t graph_size = 0;
    if(can_use_scatter())
    {
        for (int sh = 0; sh < shards_number; sh++)
        {
            graph_size += outgoing_shards[sh].get_size();
        }
    }
    if(can_use_gather())
    {
        for (int sh = 0; sh < shards_number; sh++)
        {
            graph_size += incoming_shards[sh].get_size();
        }
    }
    graph_size += this->vertices_count * sizeof(vertices_reorder_buffer[0]);
    return graph_size;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ShardedCSRGraph::print_size()
{
    cout << "Wall size (ShardedCSRGraph): " << get_size()/1e9 << " GB" << endl;

    if(outgoing_is_stored())
    {
        for(int shard_id = 0; shard_id < shards_number; shard_id++)
        {
            cout << shard_id << ": " << 1.0*outgoing_shards[shard_id].get_non_zero_degree_vertices_count() / this->vertices_count << " (outgoing)" << endl;
        }
    }

    if(incoming_is_stored())
    {
        for(int shard_id = 0; shard_id < shards_number; shard_id++)
        {
            cout << shard_id << ": " << 1.0*incoming_shards[shard_id].get_non_zero_degree_vertices_count() / this->vertices_count  << " (incoming)" << endl;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
