#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int VectorCSRGraph::reorder_to_original(int _vertex_id)
{
    return backward_conversion[_vertex_id];
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int VectorCSRGraph::reorder_to_sorted(int _vertex_id)
{
    return forward_conversion[_vertex_id];
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VectorCSRGraph::reorder_to_original(char *_data, char *_buffer, size_t _elem_size)
{
    #if defined(__USE_GPU__)
    //cuda_reorder_wrapper_scatter(_data, _buffer, backward_conversion, this->vertices_count);
    #else
    if(_elem_size == sizeof(float))
        openmp_reorder_gather((float*)_data, (float*)_buffer, forward_conversion, this->vertices_count);
    else if(_elem_size == sizeof(double))
        openmp_reorder_gather((double*)_data, (double*)_buffer, forward_conversion, this->vertices_count);
    else
        throw "Error: incorrect element size in VectorCSRGraph::reorder_to_original";
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VectorCSRGraph::reorder_to_sorted(char *_data, char *_buffer, size_t _elem_size)
{
    #if defined(__USE_GPU__)
    //cuda_reorder_wrapper_scatter(_data, _buffer, forward_conversion, this->vertices_count);
    #else
    if(_elem_size == sizeof(float))
        openmp_reorder_gather((float*)_data, (float*)_buffer, backward_conversion, this->vertices_count);
    else if(_elem_size == sizeof(double))
        openmp_reorder_gather((double*)_data, (double*)_buffer, backward_conversion, this->vertices_count);
    else
        throw "Error: incorrect element size in VectorCSRGraph::reorder_to_sorted";
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VectorCSRGraph::reorder_edges_gather(char *_src, char *_dst, size_t _elem_size)
{
    #if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
    if(_elem_size == sizeof(float))
         openmp_reorder_gather_inplace((float*)_src, (float*)_dst, edges_reorder_indexes, this->edges_count);
    else if(_elem_size == sizeof(double))
         openmp_reorder_gather_inplace((double*)_src, (double*)_dst, edges_reorder_indexes, this->edges_count);
    else
        throw "Error: incorrect element size in VectorCSRGraph::reorder_edges_gather";
    #endif

    #if defined(__USE_GPU__)
    cuda_reorder_gather_inplace(_src, _dst, edges_reorder_indexes, this->edges_count);
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VectorCSRGraph::reorder_edges_scatter(char *_src, char *_dst, size_t _elem_size)
{
    #if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
    if(_elem_size == sizeof(float))
         openmp_reorder_scatter_inplace((float*)_src, (float*)_dst, edges_reorder_indexes, this->edges_count);
    else if(_elem_size == sizeof(double))
         openmp_reorder_scatter_inplace((double*)_src, (double*)_dst, edges_reorder_indexes, this->edges_count);
    else
        throw "Error: incorrect element size in VectorCSRGraph::reorder_edges_scatter";
    #endif

    #if defined(__USE_GPU__)
    cuda_reorder_scatter_inplace(_src, _dst, edges_reorder_indexes, this->edges_count);
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


