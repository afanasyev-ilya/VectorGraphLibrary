#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void ShardedCSRGraph::reorder_from_original_to_shard(VerticesArray<_T> _data, TraversalDirection _direction, int _shard_id)
{
    Timer tm;
    tm.start();
    if(_data.get_direction() == ORIGINAL)
    {
        if((_direction == SCATTER) && can_use_scatter())
            outgoing_shards[_shard_id].reorder_to_sorted(_data.get_ptr(), (_T*)vertices_reorder_buffer);
        else if((_direction == GATHER) && can_use_scatter())
            incoming_shards[_shard_id].reorder_to_sorted(_data.get_ptr(), (_T*)vertices_reorder_buffer);
    }
    else
    {
        throw "Error in ShardedCSRGraph::reorder_to_sorted_for_shard : unsupported direction of _data";
    }
    tm.end();
    performance_stats.update_reorder_time(tm);
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_bandwidth_stats("vertices reorder", this->vertices_count, sizeof(_T)*2 + sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void ShardedCSRGraph::reorder_from_shard_to_original(VerticesArray<_T> _data, TraversalDirection _direction, int _shard_id)
{
    Timer tm;
    tm.start();
    if(_data.get_direction() == ORIGINAL)
    {
        if((_direction == SCATTER) && can_use_scatter())
            outgoing_shards[_shard_id].reorder_to_original(_data.get_ptr(), (_T*)vertices_reorder_buffer);
        else if((_direction == GATHER) && can_use_scatter())
            incoming_shards[_shard_id].reorder_to_original(_data.get_ptr(), (_T*)vertices_reorder_buffer);
    }
    else
    {
        throw "Error in ShardedCSRGraph::reorder_to_original_for_shard : unsupported direction of _data";
    }
    tm.end();
    performance_stats.update_reorder_time(tm);
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_bandwidth_stats("vertices reorder", this->vertices_count, sizeof(_T)*2 + sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int ShardedCSRGraph::reorder(int _vertex_id, TraversalDirection _input_dir, TraversalDirection _output_dir)
{
    if((_vertex_id < 0) || (_vertex_id >= this->vertices_count))
        throw "Error in ShardedCSRGraph::reorder : _vertex_id is out of range";
    if(_input_dir == ORIGINAL)
    {
        if(_output_dir == GATHER)
        {
            return incoming_shards[0].reorder_to_sorted(_vertex_id);
        }
        if(_output_dir == SCATTER)
        {
            return outgoing_shards[0].reorder_to_sorted(_vertex_id);
        }
    }
    if(_input_dir == SCATTER)
    {
        if(_output_dir == ORIGINAL)
        {
            return outgoing_shards[0].reorder_to_original(_vertex_id);
        }
        if(_output_dir == GATHER)
        {
            return incoming_shards[0].reorder_to_sorted(outgoing_shards[0].reorder_to_original(_vertex_id));
        }
    }
    if(_input_dir == GATHER)
    {
        if(_output_dir == ORIGINAL)
        {
            return incoming_shards[0].reorder_to_original(_vertex_id);
        }
        if(_output_dir == SCATTER)
        {
            return outgoing_shards[0].reorder_to_sorted(incoming_shards[0].reorder_to_original(_vertex_id));
        }
    }
    return -1;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
