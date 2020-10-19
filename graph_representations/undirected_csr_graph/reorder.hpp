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
    #if defined(__USE_NEC_SX_AURORA__) || defined(__USE_INTEL__)
    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma omp parallel for
    for(int i = 0; i < this->vertices_count; i++)
    {
        int sorted_index = i;
        int original_index = backward_conversion[i];
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
    cuda_reorder_wrapper(_data, _buffer, backward_conversion, this->vertices_count);
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void UndirectedCSRGraph::reorder_to_sorted(_T *_data, _T *_buffer)
{
    #if defined(__USE_NEC_SX_AURORA__) || defined(__USE_INTEL__)
    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma omp parallel for
    for(int i = 0; i < this->vertices_count; i++)
    {
        int original_index = i;
        int sorted_index = forward_conversion[i];
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
    cuda_reorder_wrapper(_data, _buffer, forward_conversion, this->vertices_count);
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
