#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ShardedCSRGraph::reorder_to_sorted_for_shard(VerticesArrayContainer &_container,
                                                  int _shard_id,
                                                  TraversalDirection _direction)
{
    Timer tm;
    tm.start();

    if(_container.get_direction() == ORIGINAL)
    {
        if(_container.get_element_size() == sizeof(int))
        {
            int *data = (int *)_container.get_ptr();
            if(_direction == SCATTER)
                outgoing_shards[_shard_id].reorder_to_sorted(data, (int*)vertices_reorder_buffer);
            else if(_direction == GATHER)
                incoming_shards[_shard_id].reorder_to_sorted(data, (int*)vertices_reorder_buffer);
        }
        else if(_container.get_element_size() == sizeof(long long))
        {
            long long *data = (long long *)_container.get_ptr();
            if(_direction == SCATTER)
                outgoing_shards[_shard_id].reorder_to_sorted(data, (long long*)vertices_reorder_buffer);
            else if(_direction == GATHER)
                incoming_shards[_shard_id].reorder_to_sorted(data, (long long*)vertices_reorder_buffer);
        }
        else
        {
            throw "Error in ShardedCSRGraph::reorder_to_sorted_for_shard : unsupported element size";
        }
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

void ShardedCSRGraph::reorder_to_original_for_shard(VerticesArrayContainer &_container,
                                                    int _shard_id,
                                                    TraversalDirection _direction)
{
    Timer tm;
    tm.start();
    if(_container.get_direction() == ORIGINAL)
    {
        if(_container.get_element_size() == sizeof(int))
        {
            int *data = (int *)_container.get_ptr();
            if(_direction == SCATTER)
                outgoing_shards[_shard_id].reorder_to_original(data, (int*)vertices_reorder_buffer);
            else if(_direction == GATHER)
                incoming_shards[_shard_id].reorder_to_original(data, (int*)vertices_reorder_buffer);
        }
        else if(_container.get_element_size() == sizeof(long long))
        {
            long long *data = (long long *)_container.get_ptr();
            if(_direction == SCATTER)
                outgoing_shards[_shard_id].reorder_to_original(data, (long long*)vertices_reorder_buffer);
            else if(_direction == GATHER)
                incoming_shards[_shard_id].reorder_to_original(data, (long long*)vertices_reorder_buffer);
        }
        else
        {
            throw "Error in ShardedCSRGraph::reorder_to_original_for_shard : unsupported element size";
        }
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
