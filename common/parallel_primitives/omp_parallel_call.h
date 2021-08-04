#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define OMP_PARALLEL_INTERNAL _Pragma("omp parallel")
#define OMP_BARRIER_INTERNAL _Pragma("omp barrier")

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define OMP_PARALLEL_CALL(function) \
if(omp_in_parallel())               \
{                                   \
    OMP_BARRIER_INTERNAL            \
    function;                       \
    OMP_BARRIER_INTERNAL            \
}                                   \
else                                \
{                                   \
    OMP_PARALLEL_INTERNAL           \
    {                               \
        function;                   \
    }                               \
}                                   \

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
