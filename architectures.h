#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//#define __USE_GPU__
//#define __USE_INTEL__
#define __USE_NEC_SX_AURORA__

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SX-Aurora properties
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//#define __USE_ASL__
#define VECTOR_LENGTH 256
#define LLC_CACHE_SIZE 8*1024*1024
#define MAX_SX_AURORA_THREADS 8

#define CACHED_VERTICES 3500 //4608 //4500
#define CACHE_STEP 7

#define NEC_VECTOR_ENGINE_THRESHOLD_VALUE  VECTOR_LENGTH * MAX_SX_AURORA_THREADS * 16
#define NEC_VECTOR_CORE_THRESHOLD_VALUE    4*VECTOR_LENGTH

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU properties
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define BLOCK_SIZE 1024
#define WARP_SIZE 32
#define TYPICAL_SM_COUNT 128

#define GPU_GRID_THRESHOLD_VALUE     TYPICAL_SM_COUNT * BLOCK_SIZE
#define GPU_BLOCK_THRESHOLD_VALUE    BLOCK_SIZE
#define GPU_WARP_THRESHOLD_VALUE     2*WARP_SIZE

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Common properties
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//#define __PRINT_API_PERFORMANCE_STATS__
#define __PRINT_SAMPLES_PERFORMANCE_STATS__
#define __USE_WEIGHTED_GRAPHS__

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
