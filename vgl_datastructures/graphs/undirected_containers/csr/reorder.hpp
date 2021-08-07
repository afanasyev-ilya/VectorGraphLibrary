#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSRGraph::reorder_edges_gather(char *_src, char *_dst, size_t _elem_size)
{
    #if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
    if(_elem_size == sizeof(float))
         openmp_reorder_gather_inplace((float*)_src, (float*)_dst, edges_reorder_indexes, this->edges_count);
    else if(_elem_size == sizeof(double))
         openmp_reorder_gather_inplace((double*)_src, (double*)_dst, edges_reorder_indexes, this->edges_count);
    else
        throw "Error: incorrect element size in CSRGraph::reorder_edges_gather";
    #endif

    #if defined(__USE_GPU__)
    cuda_reorder_gather_inplace(_src, _dst, edges_reorder_indexes, this->edges_count);
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSRGraph::reorder_edges_scatter(char *_src, char *_dst, size_t _elem_size)
{
    #if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
    if(_elem_size == sizeof(float))
         openmp_reorder_scatter_inplace((float*)_src, (float*)_dst, edges_reorder_indexes, this->edges_count);
    else if(_elem_size == sizeof(double))
         openmp_reorder_scatter_inplace((double*)_src, (double*)_dst, edges_reorder_indexes, this->edges_count);
    else
        throw "Error: incorrect element size in CSRGraph::reorder_edges_scatter";
    #endif

    #if defined(__USE_GPU__)
    cuda_reorder_scatter_inplace(_src, _dst, edges_reorder_indexes, this->edges_count);
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
