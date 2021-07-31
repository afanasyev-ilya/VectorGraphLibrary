#pragma once

#include "count_if/count_if.h"
#include "copy_if/copy_if.h"
#include "reorder/openmp_reorder.h"

#ifdef __USE_GPU__
#include "reorder/cuda_reorder.cuh"
#endif