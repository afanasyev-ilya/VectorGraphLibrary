#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "architectures.h"
#include "framework_types.h"
#include "architecture_independent_api.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <omp.h>
#include <string>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#ifdef __USE_MPI__
#include <mpi.h>
#endif

#ifdef __USE_NEC_SX_AURORA__
#include <ftrace.h>
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
#include <cuda_runtime.h>
#include "common/gpu_API/cuda_error_handling.h"
#include "common/gpu_API/cuda_reorder.cuh"
#include "common/gpu_API/select_device.cuh"
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "common/timer/timer.h"
#include "common/memory_API/memory_API.h"
#include "common/parallel_primitives/copy_if.h"
#include "common/performance_stats/performance_stats.h"
#include "common/library_data/library_data.h"
#include "common/random_generator/random_generator.h"
#include "common/sorter/sorter.h"
#include "common/cmd_parser/cmd_parser.h"
#include "common/parallel_primitives/get_elements_count.h"
#include "common/misc/extensions.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_analytics/graph_analytics.h"
#include "graph_generation_API/graph_generation_API.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// main VGL part: graphs in different formats
#include "graph_representations/base_graph.h"
#include "graph_representations/undirected_containers/undirected_graph.h"
#include "graph_representations/undirected_containers/edges_list/edges_list_graph.h"
#include "graph_representations/undirected_containers/vect_csr/vect_csr_graph.h"
//#include "graph_representations/undirected_containers/csr/csr_graph.h"

#include "graph_representations/vgl_graph/vgl_graph.h"

// TODO old
//#include "graph_representations/vect_csr_graph/vect_csr_graph.h"
//#include "graph_representations/sharded_csr_graph/sharded_csr_graph.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// main VGL part: computational and data abstractions
#include "graph_processing_API/common/vertices_array/vertices_array.h"
//#include "graph_processing_API/common/edges_array/edges_array.h" // TODO ACTIVATE
#include "graph_processing_API/common/frontier/frontier.h"
#include "graph_processing_API/common/graph_abstractions/graph_abstractions.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// algorithm implementations

#include "algorithms/sswp/widest_paths.h" // since vect max

#ifndef __USE_GPU__
#include "algorithms/bfs/bfs.h"
#include "algorithms/sssp/shortest_paths.h"
#include "algorithms/pr/pr.h"
#include "algorithms/cc/cc.h"
#include "algorithms/rw/random_walk.h"
#include "algorithms/hits/hits.h"
#include "algorithms/scc/scc.h"
#include "algorithms/coloring/coloring.h"
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//#include "export_graphs/ligra_export.h"
//#include "export_graphs/edges_list_export.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "common/verify_results/verify_results.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_analytics/graph_analytics.hpp"*/

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




