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
void VectCSRGraph::reorder_to_original(VerticesArrayNec<_T> &_data)
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
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_bandwidth_stats("vertices reorder", this->vertices_count, sizeof(_T)*2 + sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void VectCSRGraph::reorder_to_scatter(VerticesArrayNec<_T> &_data)
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
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_bandwidth_stats("vertices reorder", this->vertices_count, sizeof(_T)*2 + sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void VectCSRGraph::reorder_to_gather(VerticesArrayNec<_T> &_data)
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
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_bandwidth_stats("vertices reorder", this->vertices_count, sizeof(_T)*2 + sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void VectCSRGraph::reorder(VerticesArrayNec<_T> &_data, TraversalDirection _output_dir)
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

void VectCSRGraph::reorder(FrontierNEC &_data, TraversalDirection _output_dir)
{
    throw "Error in VectCSRGraph::reorder : can not reorder a frontier";
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void VectCSRGraph::reorder_edges_to_gather(_T *_incoming_csr_ptr, _T *_outgoing_csr_ptr)
{
    Timer tm;
    tm.start();

    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma omp parallel for
    for(long long i = 0; i < this->edges_count; i++)
    {
        _incoming_csr_ptr[i] = _outgoing_csr_ptr[edges_reorder_indexes[i]];
    }

    tm.end();
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_bandwidth_stats("vertices reorder", this->vertices_count, sizeof(_T)*2 + sizeof(_outgoing_csr_ptr[0]));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
bool VectCSRGraph::vertices_buffer_can_be_used(VerticesArrayNec<_T> &_data)
{
    if(sizeof(_T) <= sizeof(vertices_reorder_buffer[0]))
        return true;
    else
        return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


