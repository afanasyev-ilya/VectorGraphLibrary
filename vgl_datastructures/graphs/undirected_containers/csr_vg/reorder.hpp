#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSR_VG_Graph::reorder_edges_gather(char *_src, char *_dst, size_t _elem_size)
{
    #if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
    if(_elem_size == sizeof(float))
         openmp_reorder_gather_copy((float*)_src, (float*)_dst, edges_reorder_indexes, this->edges_count);
    else if(_elem_size == sizeof(double))
         openmp_reorder_gather_copy((double*)_src, (double*)_dst, edges_reorder_indexes, this->edges_count);
    else
        throw "Error: incorrect element size in CSR_VG_Graph::reorder_edges_gather";
    #endif

    #if defined(__USE_GPU__)
    if(_elem_size == sizeof(float))
         cuda_reorder_gather_copy((float*)_src, (float*)_dst, edges_reorder_indexes, this->edges_count);
    else if(_elem_size == sizeof(double))
         cuda_reorder_gather_copy((double*)_src, (double*)_dst, edges_reorder_indexes, this->edges_count);
    else
        throw "Error: incorrect element size in CSR_VG_Graph::reorder_edges_gather";
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSR_VG_Graph::reorder_edges_scatter(char *_src, char *_dst, size_t _elem_size)
{
    #if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
    if(_elem_size == sizeof(float))
         openmp_reorder_scatter_copy((float*)_src, (float*)_dst, edges_reorder_indexes, this->edges_count);
    else if(_elem_size == sizeof(double))
         openmp_reorder_scatter_copy((double*)_src, (double*)_dst, edges_reorder_indexes, this->edges_count);
    else
        throw "Error: incorrect element size in CSR_VG_Graph::reorder_edges_scatter";
    #endif

    #if defined(__USE_GPU__)
    if(_elem_size == sizeof(float))
         cuda_reorder_scatter_copy((float*)_src, (float*)_dst, edges_reorder_indexes, this->edges_count);
    else if(_elem_size == sizeof(double))
         cuda_reorder_scatter_copy((double*)_src, (double*)_dst, edges_reorder_indexes, this->edges_count);
    else
        throw "Error: incorrect element size in CSR_VG_Graph::reorder_edges_scatter";
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
