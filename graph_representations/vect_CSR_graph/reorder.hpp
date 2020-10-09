#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int VectCSRGraph::reorder(int _vertex_id, TraversalDirection _input_dir, TraversalDirection _output_dir)
{
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
            throw incoming_graph->reorder_to_sorted(outgoing_graph->reorder_to_original(_vertex_id));
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
            throw outgoing_graph->reorder_to_sorted(incoming_graph->reorder_to_original(_vertex_id));
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

    _T *buffer;
    MemoryAPI::allocate_array(&buffer, this->vertices_count);

    // allocate buffer if not enough space
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

    _T *buffer;
    MemoryAPI::allocate_array(&buffer, this->vertices_count);

    // allocate buffer if not enough space
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

    _data.set_direction(ORIGINAL);

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

    _T *buffer;
    MemoryAPI::allocate_array(&buffer, this->vertices_count);

    // allocate buffer if not enough space
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

    _data.set_direction(ORIGINAL);

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
        _incoming_csr_ptr[i] = _outgoing_csr_ptr[edges_reorder_buffer[i]];
    }

    tm.end();
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_bandwidth_stats("vertices reorder", this->vertices_count, sizeof(_T)*2 + sizeof(_outgoing_csr_ptr[0]));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

