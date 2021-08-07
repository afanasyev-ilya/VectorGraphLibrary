#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void CSRGraph::reorder_edges_gather(_T *_src, _T *_dst)
{
    #if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
    openmp_reorder_gather_inplace(_src, _dst, edges_reorder_indexes, this->edges_count);
    #endif

    #if defined(__USE_GPU__)
    cuda_reorder_gather_inplace(_src, _dst, edges_reorder_indexes, this->edges_count);
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void CSRGraph::reorder_edges_scatter(_T *_src, _T *_dst)
{
    #if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
    openmp_reorder_scatter_inplace(_src, _dst, edges_reorder_indexes, this->edges_count);
    #endif

    #if defined(__USE_GPU__)
    cuda_reorder_scatter_inplace(_src, _dst, edges_reorder_indexes, this->edges_count);
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
