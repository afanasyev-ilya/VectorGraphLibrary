#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// optimization settings
#define __USE_CSR_VERTEX_GROUPS__
//#define __SORT_ADJACENT_EDGES__

#ifdef __USE_CSR_VERTEX_GROUPS__
#define CSR_VERTEX_GROUPS_NUM 6
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
//#warning "NEC SX-Aurora TSUBASA API is used!"
#pragma message("NEC SX-Aurora TSUBASA API is used!")
#endif

#ifdef __USE_MULTICORE__
//#warning "Multicore API is used!"
#pragma message("Multicore API is used!")
#endif

#ifdef __USE_GPU__
//#warning "GPU API is used!"
#pragma message("GPU API is used!")
#endif

#ifdef __USE_CSR_VERTEX_GROUPS__
//#warning "NEC SX-Aurora TSUBASA API is used!"
#pragma message("CSR vertex groups are used!")
#endif


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SX-Aurora properties
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
#define __USE_ASL__
#define VECTOR_LENGTH 256
#define VECTOR_LENGTH_POW 8
#define MAX_SX_AURORA_THREADS 8
#define LLC_CACHE_SIZE 16*1024*1024
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU properties
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
//#define _GLIBCXX_PARALLEL
#define VECTOR_LENGTH 32
#define VECTOR_LENGTH_POW 5
#define MAX_SX_AURORA_THREADS 8
#define LLC_CACHE_SIZE 6*1024*1024
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ARM/Intel/AMD multicore properties
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_MULTICORE__
#define _GLIBCXX_PARALLEL
#define VECTOR_LENGTH 32
#define VECTOR_LENGTH_POW 5
#define MAX_SX_AURORA_THREADS 256
#define LLC_CACHE_SIZE 512*1024*1024
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU properties
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define BLOCK_SIZE 1024
#define WARP_SIZE 32
#define TYPICAL_SM_COUNT 128

#define GPU_GRID_THRESHOLD_VALUE     TYPICAL_SM_COUNT * BLOCK_SIZE
#define GPU_BLOCK_THRESHOLD_VALUE    BLOCK_SIZE
#define GPU_WARP_THRESHOLD_VALUE     WARP_SIZE

#define __USE_MANAGED_MEMORY__

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Common properties
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define MAX_WEIGHT 100

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// main framework settings
// define if user hasn't set any algorithm-specific
#ifndef VECTOR_ENGINE_THRESHOLD_VALUE
#ifdef __USE_GPU__
#define VECTOR_ENGINE_THRESHOLD_VALUE BLOCK_SIZE
#else
#define VECTOR_ENGINE_THRESHOLD_VALUE VECTOR_LENGTH * MAX_SX_AURORA_THREADS * 128
#endif
#endif

#ifndef VECTOR_CORE_THRESHOLD_VALUE
#define VECTOR_CORE_THRESHOLD_VALUE VECTOR_LENGTH
#endif

#ifndef FRONTIER_TYPE_CHANGE_THRESHOLD
#define FRONTIER_TYPE_CHANGE_THRESHOLD 0.3
#endif

#ifndef VE_FRONTIER_TYPE_CHANGE_THRESHOLD
#define VE_FRONTIER_TYPE_CHANGE_THRESHOLD 0.01
#endif

#ifndef VC_FRONTIER_TYPE_CHANGE_THRESHOLD
#define VC_FRONTIER_TYPE_CHANGE_THRESHOLD 0.01
#endif

#ifndef COLLECTIVE_FRONTIER_TYPE_CHANGE_THRESHOLD
#define COLLECTIVE_FRONTIER_TYPE_CHANGE_THRESHOLD 0.15
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define __PRINT_API_PERFORMANCE_STATS__ // prints inner api detailed performance stats, causes significant delays when active
#define __PRINT_SAMPLES_PERFORMANCE_STATS__ // prints samples stats (iterations, bandwidths, components stats)
//#define __SAVE_PERFORMANCE_STATS_TO_FILE__ // saves performance stats to files (useful for multiple batch launches)

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define VISUALISATION_SMALL_GRAPH_VERTEX_THRESHOLD 30

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Variables, used for API performance stats

#ifndef INT_ELEMENTS_PER_EDGE
#define INT_ELEMENTS_PER_EDGE 3.0
#endif

#ifndef COMPUTE_INT_ELEMENTS
#define COMPUTE_INT_ELEMENTS 2.0
#endif

#ifndef REDUCE_INT_ELEMENTS
#define REDUCE_INT_ELEMENTS 2.0
#endif

#ifndef GNF_INT_ELEMENTS
#define GNF_INT_ELEMENTS 1.0
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


