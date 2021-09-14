#pragma once

#if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
#define __VGL_COMPUTE_ARGS__ (int src_id, int connections_count, int vector_index)
#define __VGL_SCATTER_ARGS__ (int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos, int vector_index)
#define __VGL_GATHER_ARGS__ (int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos, int vector_index)
#define __VGL_ADVANCE_ARGS__ (int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos, int vector_index)
#define __VGL_ADVANCE_PREPROCESS_ARGS__ (int src_id, int connections_count, int vector_index)
#define __VGL_ADVANCE_POSTPROCESS_ARGS__ (int src_id, int connections_count, int vector_index)
#define __VGL_GNF_ARGS__ (int src_id, int connections_count)->int
#define __VGL_REDUCE_ANY_ARGS__ (int src_id, int connections_count, int vector_index)
#define __VGL_REDUCE_INT_ARGS__ (int src_id, int connections_count, int vector_index)->int
#define __VGL_REDUCE_FLT_ARGS__ (int src_id, int connections_count, int vector_index)->float
#define __VGL_REDUCE_DBL_ARGS__ (int src_id, int connections_count, int vector_index)->double
#define __VGL_COPY_IF_INDEXES_ARGS__ (int idx)
#endif

#ifdef __USE_GPU__
#define __VGL_COMPUTE_ARGS__ __device__ (int src_id, int connections_count, int vector_index)
#define __VGL_SCATTER_ARGS__ __device__ (int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos, int vector_index)
#define __VGL_GATHER_ARGS__ __device__ (int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos, int vector_index)
#define __VGL_ADVANCE_ARGS__ __device__ (int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos, int vector_index)
#define __VGL_ADVANCE_PREPROCESS_ARGS__ __device__ (int src_id, int connections_count, int vector_index)
#define __VGL_ADVANCE_POSTPROCESS_ARGS__ __device__ (int src_id, int connections_count, int vector_index)
#define __VGL_GNF_ARGS__ __device__ (int src_id, int connections_count)->int
#define __VGL_REDUCE_ANY_ARGS__ __device__ (int src_id, int connections_count, int vector_index)
#define __VGL_REDUCE_INT_ARGS__ __device__ (int src_id, int connections_count, int vector_index)->int
#define __VGL_REDUCE_FLT_ARGS__ __device__ (int src_id, int connections_count, int vector_index)->float
#define __VGL_REDUCE_DBL_ARGS__ __device__ (int src_id, int connections_count, int vector_index)->double
#define __VGL_COPY_IF_INDEXES_ARGS__ __device__ (int idx)
#endif

#ifdef __USE_NEC_SX_AURORA__
#define VGL_GRAPH_ABSTRACTIONS GraphAbstractionsNEC
#endif

#ifdef __USE_MULTICORE__
#define VGL_GRAPH_ABSTRACTIONS GraphAbstractionsMulticore
#endif

#ifdef __USE_GPU__
#define VGL_GRAPH_ABSTRACTIONS GraphAbstractionsGPU
#endif

#define VGL_FRONTIER VGL_Frontier

#ifdef __USE_GPU__
#define VGL_SRC_ID_ADD(a, b) (atomicAdd(&(a), (b)))
#else
#define VGL_SRC_ID_ADD(a, b) ((a) += (b))
#endif

#ifdef __USE_GPU__
#define VGL_INC(a) (atomicAdd(&(a), 1))
#else
#define VGL_INC(a) ((a) += 1)
#endif

#ifdef __USE_GPU__
#define VGL_DEC(a) (atomicSub(&(a), 1))
#else
#define VGL_DEC(a) ((a) += 1)
#endif