#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class ParallelPrimitives
{
private:
    template <typename CopyCondition>
    static inline int vector_copy_if_indexes(CopyCondition &&_cond,
                                             int *_out_data,
                                             size_t _size,
                                             int *_buffer,
                                             const int _buffer_size,
                                             const int _index_offset);

    template <typename CopyCondition>
    static inline int omp_copy_if_indexes(CopyCondition &&_cond,
                                          int *_out_data,
                                          size_t _size,
                                          int *_buffer,
                                          const int _buffer_size,
                                          const int _index_offset);

    template <typename CopyCondition, typename _T>
    static inline int omp_copy_if_data(CopyCondition &&_cond,
                                       _T *_in_data,
                                       _T *_out_data,
                                       size_t _size,
                                       _T *_buffer,
                                       const int _buffer_size);
public:
    template <typename CopyCondition>
    static inline int copy_if_indexes(CopyCondition &&_cond,
                                      int *_out_data,
                                      size_t _size,
                                      int *_buffer,
                                      const int _buffer_size,
                                      const int _index_offset);

    template <typename CopyCondition, typename _T>
    static inline int copy_if_data(CopyCondition &&_cond,
                                   _T *_in_data,
                                   _T *_out_data,
                                   size_t _size,
                                   _T *_buffer,
                                   const int _buffer_size);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "count_if/count_if.h"
#include "copy_if/copy_if.hpp"
#include "reorder/openmp_reorder.h"
#include "omp_parallel_call.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
#include "reorder/cuda_reorder.cuh"
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

