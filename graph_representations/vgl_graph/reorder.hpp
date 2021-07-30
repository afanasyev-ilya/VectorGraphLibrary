/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void VGL_Graph::reorder(VerticesArray<_T> &_data, TraversalDirection _output_dir)
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
void VGL_Graph::reorder_to_original(VerticesArray<_T> &_data)
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
        outgoing_data->reorder_to_original((char*)_data.get_ptr(), (char*)buffer, sizeof(_data[0]));
    }
    else if(_data.get_direction() == GATHER)
    {
        incoming_data->reorder_to_original((char*)_data.get_ptr(), (char*)buffer, sizeof(_data[0]));
    }

    _data.set_direction(ORIGINAL);

    if(buffer_was_allocated)
        MemoryAPI::free_array(buffer);

    tm.end();
    performance_stats.update_reorder_time(tm);
    performance_stats.update_bytes_requested((sizeof(_T)*2 + sizeof(int))*this->vertices_count);

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_bandwidth_stats("vertices reorder", this->vertices_count, sizeof(_T)*2 + sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void VGL_Graph::reorder_to_scatter(VerticesArray<_T> &_data)
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
        outgoing_data->reorder_to_sorted((char*)_data.get_ptr(), (char*)buffer, sizeof(_data[0]));
    }
    else if(_data.get_direction() == GATHER)
    {
        incoming_data->reorder_to_original((char*)_data.get_ptr(), (char*)buffer, sizeof(_data[0]));
        outgoing_data->reorder_to_sorted((char*)_data.get_ptr(), (char*)buffer, sizeof(_data[0]));
    }

    _data.set_direction(SCATTER);

    if(buffer_was_allocated)
        MemoryAPI::free_array(buffer);

    tm.end();
    performance_stats.update_reorder_time(tm);
    performance_stats.update_bytes_requested((sizeof(_T)*2 + sizeof(int))*this->vertices_count);

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_bandwidth_stats("vertices reorder", this->vertices_count, sizeof(_T)*2 + sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void VGL_Graph::reorder_to_gather(VerticesArray<_T> &_data)
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
        incoming_data->reorder_to_sorted((char*)_data.get_ptr(), (char*)buffer, sizeof(_data[0]));
    }
    else if(_data.get_direction() == SCATTER)
    {
        outgoing_data->reorder_to_original((char*)_data.get_ptr(), (char*)buffer, sizeof(_data[0]));
        incoming_data->reorder_to_sorted((char*)_data.get_ptr(), (char*)buffer, sizeof(_data[0]));
    }

    _data.set_direction(GATHER);

    if(buffer_was_allocated)
        MemoryAPI::free_array(buffer);

    tm.end();
    performance_stats.update_reorder_time(tm);
    performance_stats.update_bytes_requested((sizeof(_T)*2 + sizeof(int))*this->vertices_count);

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_bandwidth_stats("vertices reorder", this->vertices_count, sizeof(_T)*2 + sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
bool VGL_Graph::vertices_buffer_can_be_used(VerticesArray<_T> &_data)
{
    if(sizeof(_T) <= sizeof(vertices_reorder_buffer[0]))
        return true;
    else
        return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int VGL_Graph::reorder(int _vertex_id, TraversalDirection _input_dir, TraversalDirection _output_dir)
{
    if((_vertex_id < 0) || (_vertex_id >= this->vertices_count))
        throw "Error in VGL_Graph::reorder : _vertex_id is out of range";

    if(_input_dir == ORIGINAL)
    {
        if(_output_dir == GATHER)
        {
            return incoming_data->reorder_to_sorted(_vertex_id);
        }
        if(_output_dir == SCATTER)
        {
            return outgoing_data->reorder_to_sorted(_vertex_id);
        }
    }
    if(_input_dir == SCATTER)
    {
        if(_output_dir == ORIGINAL)
        {
            return outgoing_data->reorder_to_original(_vertex_id);
        }
        if(_output_dir == GATHER)
        {
            return incoming_data->reorder_to_sorted(outgoing_data->reorder_to_original(_vertex_id));
        }
    }
    if(_input_dir == GATHER)
    {
        if(_output_dir == ORIGINAL)
        {
            return incoming_data->reorder_to_original(_vertex_id);
        }
        if(_output_dir == SCATTER)
        {
            return outgoing_data->reorder_to_sorted(incoming_data->reorder_to_original(_vertex_id));
        }
    }
    return -1;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int VGL_Graph::select_random_vertex(TraversalDirection _direction)
{
    return reorder(outgoing_data->select_random_vertex(), SCATTER, _direction);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
