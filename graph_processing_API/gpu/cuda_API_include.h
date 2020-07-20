#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

extern double INNER_WALL_TIME;
extern double INNER_ADVANCE_TIME;
extern double DETAILED_ADVANCE_PART_1_NEC_TIME;
extern double DETAILED_ADVANCE_PART_2_NEC_TIME;
extern double DETAILED_ADVANCE_PART_3_NEC_TIME;
extern double INNER_COMPUTE_TIME;
extern double INNER_GNF_TIME;
extern double INNER_FILTER_TIME;
extern double INNER_REDUCE_TIME;
extern double INNER_PACK_TIME;
extern double INNER_WALL_WORK;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <omp.h>
#include <iostream>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <ctime>
#include "../../common/gpu_API/cuda_error_handling.h"
#include "../../architectures.h"
#include <cfloat>
#include <cuda_fp16.h>
#include "../../graph_representations/base_graph.h"
#include "../../graph_representations/edges_list_graph/edges_list_graph.h"
#include "../../graph_representations/extended_CSR_graph/extended_CSR_graph.h"
#include "../../graph_processing_API/gpu/graph_primitives/graph_primitives_gpu.cuh"


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////