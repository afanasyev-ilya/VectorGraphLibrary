#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void ShardedCSRGraph::reorder_to_sorted_for_shard(char *_data, int _element_size, int _shard_id)
{
    Timer tm;
    tm.start();

    if(_data.get_direction() == ORIGINAL)
    {
        if(_element_size == sizeof(int))
        {
            int *data = (int *) _data;
            outgoing_shards[_shard_id].reorder_to_sorted(_data, (int*)vertices_reorder_buffer);
        }
        else if(_element_size == sizeof(long long))
        {
            long long *data = (long long *) _data;
            outgoing_shards[_shard_id].reorder_to_sorted(_data, (*)vertices_reorder_buffer);
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

template <typename _T>
void ShardedCSRGraph::reorder_to_original_for_shard(VerticesArray<_T> _data, int _shard_id)
{
    Timer tm;
    tm.start();
    if(_data.get_direction() == ORIGINAL)
    {
        outgoing_shards[_shard_id].reorder_to_original(_data.get_ptr(), (_T*)vertices_reorder_buffer);
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
