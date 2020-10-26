#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int VectCSRGraph::reorder(int _vertex_id, TraversalDirection _input_dir, TraversalDirection _output_dir)
{
    if((_vertex_id < 0) || (_vertex_id >= this->vertices_count))
        throw "Error in VectCSRGraph::reorder : _vertex_id is out of range";
    if(_input_dir == ORIGINAL)
    {
        if(_output_dir == GATHER)
        {
            return incoming_graph->reorder_to_sorted(_vertex_id);
        }
        if(_output_dir == SCATTER)
        {
            return outgoing_graph->reorder_to_sorted(_vertex_id);
        }
    }
    if(_input_dir == SCATTER)
    {
        if(_output_dir == ORIGINAL)
        {
            return outgoing_graph->reorder_to_original(_vertex_id);
        }
        if(_output_dir == GATHER)
        {
            return incoming_graph->reorder_to_sorted(outgoing_graph->reorder_to_original(_vertex_id));
        }
    }
    if(_input_dir == GATHER)
    {
        if(_output_dir == ORIGINAL)
        {
            return incoming_graph->reorder_to_original(_vertex_id);
        }
        if(_output_dir == SCATTER)
        {
            return outgoing_graph->reorder_to_sorted(incoming_graph->reorder_to_original(_vertex_id));
        }
    }
    return -1;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void VectCSRGraph::reorder_to_original(VerticesArray<_T> &_data)
{
    Timer tm;
    tm.start();

    // allocate buffer if not enough space
    bool buffer_was_allocated = false;
    _T *buffer;
    if(vertices_buffer_can_be_used(_data))
    {
        buffer = (_T *) vertices_reorder_buffer;
        buffer_was_allocated = false;
    }
    else
    {
        MemoryAPI::allocate_array(&buffer, this->vertices_count);
        buffer_was_allocated = true;
    }

    // do reorder
    if(_data.get_direction() == ORIGINAL)
    {
        return;
    }
    else if(_data.get_direction() == SCATTER)
    {
        outgoing_graph->reorder_to_original(_data.get_ptr(), buffer);
    }
    else if(_data.get_direction() == GATHER)
    {
        incoming_graph->reorder_to_original(_data.get_ptr(), buffer);
    }

    _data.set_direction(ORIGINAL);

    if(buffer_was_allocated)
        MemoryAPI::free_array(buffer);

    tm.end();
    performance_stats.update_reorder_time(tm);
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_bandwidth_stats("vertices reorder", this->vertices_count, sizeof(_T)*2 + sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void VectCSRGraph::reorder_to_scatter(VerticesArray<_T> &_data)
{
    Timer tm;
    tm.start();

    // allocate buffer if not enough space
    bool buffer_was_allocated = false;
    _T *buffer;
    if(vertices_buffer_can_be_used(_data))
    {
        buffer = (_T *) vertices_reorder_buffer;
        buffer_was_allocated = false;
    }
    else
    {
        MemoryAPI::allocate_array(&buffer, this->vertices_count);
        buffer_was_allocated = true;
    }

    // do reorder
    if(_data.get_direction() == SCATTER)
    {
        return;
    }
    else if(_data.get_direction() == ORIGINAL)
    {
        outgoing_graph->reorder_to_sorted(_data.get_ptr(), buffer);
    }
    else if(_data.get_direction() == GATHER)
    {
        incoming_graph->reorder_to_original(_data.get_ptr(), buffer);
        outgoing_graph->reorder_to_sorted(_data.get_ptr(), buffer);
    }

    _data.set_direction(SCATTER);

    if(buffer_was_allocated)
        MemoryAPI::free_array(buffer);

    tm.end();
    performance_stats.update_reorder_time(tm);
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_bandwidth_stats("vertices reorder", this->vertices_count, sizeof(_T)*2 + sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void VectCSRGraph::reorder_to_gather(VerticesArray<_T> &_data)
{
    Timer tm;
    tm.start();

    // allocate buffer if not enough space
    bool buffer_was_allocated = false;
    _T *buffer;
    if(vertices_buffer_can_be_used(_data))
    {
        buffer = (_T *) vertices_reorder_buffer;
        buffer_was_allocated = false;
    }
    else
    {
        MemoryAPI::allocate_array(&buffer, this->vertices_count);
        buffer_was_allocated = true;
    }

    // do reorder
    if(_data.get_direction() == GATHER)
    {
        return;
    }
    else if(_data.get_direction() == ORIGINAL)
    {
        incoming_graph->reorder_to_sorted(_data.get_ptr(), buffer);
    }
    else if(_data.get_direction() == SCATTER)
    {
        outgoing_graph->reorder_to_original(_data.get_ptr(), buffer);
        incoming_graph->reorder_to_sorted(_data.get_ptr(), buffer);
    }

    _data.set_direction(GATHER);

    if(buffer_was_allocated)
        MemoryAPI::free_array(buffer);

    tm.end();
    performance_stats.update_reorder_time(tm);
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_bandwidth_stats("vertices reorder", this->vertices_count, sizeof(_T)*2 + sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void VectCSRGraph::reorder(VerticesArray<_T> &_data, TraversalDirection _output_dir)
{
    if(_output_dir == SCATTER)
    {
        reorder_to_scatter(_data);
    }
    if(_output_dir == GATHER)
    {
        reorder_to_gather(_data);
    }
    if(_output_dir == ORIGINAL)
    {
        reorder_to_original(_data);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void VectCSRGraph::reorder_edges_original_to_scatter(_T *_scatter_data, _T *_original_data)
{
    Timer tm;
    tm.start();

    outgoing_graph->reorder_and_copy_edges_from_original_to_sorted(_scatter_data, _original_data);

    tm.end();
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_bandwidth_stats("vertices reorder", this->vertices_count, sizeof(_T)*2);
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void VectCSRGraph::reorder_edges_scatter_to_gather(_T *_gather_data, _T *_scatter_data)
{
    Timer tm;
    tm.start();

    incoming_graph->reorder_and_copy_edges_from_original_to_sorted(_gather_data, _scatter_data);

    tm.end();
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_bandwidth_stats("vertices reorder", this->vertices_count, sizeof(_T)*2);
    #endif
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
bool VectCSRGraph::vertices_buffer_can_be_used(VerticesArray<_T> &_data)
{
    if(sizeof(_T) <= sizeof(vertices_reorder_buffer[0]))
        return true;
    else
        return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


