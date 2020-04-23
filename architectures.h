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
#define VECTOR_LENGTH_POW 8
#define MAX_SX_AURORA_THREADS 8

#define NEC_VECTOR_ENGINE_THRESHOLD_VALUE  VECTOR_LENGTH * MAX_SX_AURORA_THREADS * 128
//#define NEC_VECTOR_CORE_THRESHOLD_VALUE    3*VECTOR_LENGTH
#define NEC_VECTOR_CORE_THRESHOLD_VALUE    VECTOR_LENGTH

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU properties
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define BLOCK_SIZE 1024
#define WARP_SIZE 32
#define TYPICAL_SM_COUNT 128

#define GPU_GRID_THRESHOLD_VALUE     TYPICAL_SM_COUNT * BLOCK_SIZE
#define GPU_BLOCK_THRESHOLD_VALUE    BLOCK_SIZE
#define GPU_WARP_THRESHOLD_VALUE     WARP_SIZE

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Common properties
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define __USE_WEIGHTED_GRAPHS__

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//#define __PRINT_API_PERFORMANCE_STATS__ // prints inner api detailed performance stats, causes significant delays when active
//#define __PRINT_SAMPLES_PERFORMANCE_STATS__ // prints samples stats (iterations, bandwidths, components stats)
//#define __SAVE_PERFORMANCE_STATS_TO_FILE__ // saves performance stats to files (useful for multiple batch launches)

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define VISUALISATION_SMALL_GRAPH_VERTEX_THRESHOLD 30

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

