#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "architectures.h"
#include "framework_types.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <omp.h>
#include <string>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#ifdef __USE_GPU__
#include <cuda_runtime.h>
#include "common/gpu_API/cuda_error_handling.h"
#endif

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "common/timer/timer.h"
#include "common/memory_API/memory_API.h"
#include "common/random_generator/random_generator.h"
#include "common/sorter/sorter.h"
#include "common/cmd_parser/cmd_parser.h"
#include "common/parallel_primitives/copy_if.h"
#include "common/parallel_primitives/get_elements_count.h"
#include "common/performance_stats/performance_stats.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// main VGL part: graphs in different formats
#include "graph_representations/base_graph.h"
#include "graph_representations/edges_list_graph/edges_list_graph.h"
#include "graph_representations/undirected_csr_graph/undirected_csr_graph.h"
#include "graph_representations/vect_csr_graph/vect_csr_graph.h"
#include "graph_representations/sharded_csr_graph/sharded_csr_graph.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// main VGL part: computational and data abstractions
#include "graph_processing_API/common/vertices_array/vertices_array.h"
#include "graph_processing_API/common/edges_array/edges_array.h"
#include "graph_processing_API/common/frontier/frontier.h"
#include "graph_processing_API/common/graph_abstractions/graph_abstractions.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_generation_API/graph_generation_API.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// algorithm implementations
//#include "algorithms/bfs/bfs.h"
#include "algorithms/sssp/shortest_paths.h"
//#include "algorithms/scc/scc.h"
/*#include "algorithms/sswp/widest_paths.h"
#include "algorithms/cc/cc.h"
#include "algorithms/pr/pr.h"
#include "algorithms/lp/lp.h"
#include "algorithms/mf/mf.h"*/

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//#include "export_graphs/ligra_export.h"
//#include "export_graphs/edges_list_export.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//#include "graph_analytics/graph_analytics.h"
#include "common/verify_results/verify_results.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



