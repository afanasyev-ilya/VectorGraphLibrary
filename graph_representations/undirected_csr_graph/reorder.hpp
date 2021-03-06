#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int UndirectedCSRGraph::reorder_to_original(int _vertex_id)
{
    return backward_conversion[_vertex_id];
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int UndirectedCSRGraph::reorder_to_sorted(int _vertex_id)
{
    return forward_conversion[_vertex_id];
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void UndirectedCSRGraph::reorder_to_original(_T *_data, _T *_buffer)
{
    #if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma omp parallel for
    for(int i = 0; i < this->vertices_count; i++)
    {
        int sorted_index = forward_conversion[i];
        int original_index = i;
        _buffer[original_index] = _data[sorted_index];
    }

    #pragma _NEC ivdep
    #pragma omp parallel for
    for(int i = 0; i < this->vertices_count; i++)
    {
        _data[i] = _buffer[i];
    }
    #endif

    #if defined(__USE_GPU__)
    cuda_reorder_wrapper_scatter(_data, _buffer, backward_conversion, this->vertices_count);
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void UndirectedCSRGraph::reorder_to_sorted(_T *_data, _T *_buffer)
{
    #if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma omp parallel for
    for(int i = 0; i < this->vertices_count; i++)
    {
        int original_index = backward_conversion[i];
        int sorted_index = i;
        _buffer[sorted_index] = _data[original_index];
    }

    #pragma _NEC ivdep
    #pragma omp parallel for
    for(int i = 0; i < this->vertices_count; i++)
    {
        _data[i] = _buffer[i];
    }
    #endif

    #if defined(__USE_GPU__)
    cuda_reorder_wrapper_scatter(_data, _buffer, forward_conversion, this->vertices_count); // TODO which direction is faster?
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void UndirectedCSRGraph::update_edge_reorder_indexes_using_superposition(vgl_sort_indexes *_outer_edges_reorder_indexes)
{
    vgl_sort_indexes *buffer;
    MemoryAPI::allocate_array(&buffer, this->edges_count);
    for(long long i = 0; i < this->edges_count; i++)
    {
        buffer[i] = _outer_edges_reorder_indexes[edges_reorder_indexes[i]];
    }

    MemoryAPI::copy(edges_reorder_indexes, buffer, this->edges_count);
    MemoryAPI::free_array(buffer);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void UndirectedCSRGraph::reorder_edges_to_sorted(_T *_data, _T *_buffer)
{
    bool buffer_was_allocated = false;
    if(_buffer == NULL)
    {
        MemoryAPI::allocate_array(&_buffer, this->edges_count);
        buffer_was_allocated = true;
    }

    #if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma omp parallel for
    for(vgl_sort_indexes i = 0; i < this->edges_count; i++)
    {
        vgl_sort_indexes sorted_index = i;
        vgl_sort_indexes original_index = edges_reorder_indexes[i];
        _buffer[original_index] = _data[sorted_index];
    }

    #pragma _NEC ivdep
    #pragma omp parallel for
    for(vgl_sort_indexes i = 0; i < this->edges_count; i++)
    {
        _data[i] = _buffer[i];
    }
    #endif

    #if defined(__USE_GPU__)
    throw "Error UndirectedCSRGraph::reorder_edges_to_sorted : not implemented yet";
    #endif

    if(buffer_was_allocated)
    {
        MemoryAPI::free_array(_buffer);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void UndirectedCSRGraph::reorder_edges_to_original(_T *_data, _T *_buffer)
{
    bool buffer_was_allocated = false;
    if(_buffer == NULL)
    {
        MemoryAPI::allocate_array(&_buffer, this->edges_count);
        buffer_was_allocated = true;
    }

    #if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma omp parallel for
    for(vgl_sort_indexes i = 0; i < this->edges_count; i++)
    {
        vgl_sort_indexes sorted_index = edges_reorder_indexes[i];
        vgl_sort_indexes original_index = i;
        _buffer[original_index] = _data[sorted_index];
    }

    #pragma _NEC ivdep
    #pragma omp parallel for
    for(vgl_sort_indexes i = 0; i < this->edges_count; i++)
    {
        _data[i] = _buffer[i];
    }
    #endif

    #if defined(__USE_GPU__)
    throw "Error UndirectedCSRGraph::reorder_edges_to_original : not implemented yet";
    #endif

    if(buffer_was_allocated)
    {
        MemoryAPI::free_array(_buffer);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void UndirectedCSRGraph::reorder_and_copy_edges_from_original_to_sorted(_T *_dst_sorted, _T *_src_original)
{
    #if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma omp parallel for
    for(vgl_sort_indexes i = 0; i < this->edges_count; i++)
    {
        vgl_sort_indexes original_index = edges_reorder_indexes[i];
        _dst_sorted[i] = _src_original[original_index];
    }
    #endif

    #if defined(__USE_GPU__)
    cuda_reorder_wrapper_gather_copy(_dst_sorted, _src_original, edges_reorder_indexes, this->edges_count);
    //throw "Error UndirectedCSRGraph::reorder_and_copy_edges_from_original_to_sorted : not implemented yet";
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
