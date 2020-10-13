#pragma once

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