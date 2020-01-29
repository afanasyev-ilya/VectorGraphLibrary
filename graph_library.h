//
//  graph_library.h
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 14/04/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef graph_library_h
#define graph_library_h

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <omp.h>

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "architectures.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "common_datastructures/vector_dictionary/vector_dictionary.h"
#include "common_datastructures/random_generator/random_generation_API.h"
#include "common_datastructures/sorting_API/sorting_API.h"
#include "common_datastructures/vector_sorter/vector_sorter.h"
#include "common_datastructures/verify_results/verify_results.h"
#include "common_datastructures/cmd_parser/cmd_parser.h"
#include "common_datastructures/parallel_primitives/copy_if.h"
#include "common_datastructures/parallel_primitives/get_elements_count.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
#include "common_datastructures/gpu_API/gpu_arrays.h"
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_representations/base_graph.h"
#include "graph_representations/edges_list_graph/edges_list_graph.h"
#include "graph_representations/extended_CSR_graph/extended_CSR_graph.h"
#include "graph_representations/vectorised_CSR_graph/vectorised_CSR_graph.h"
#include "graph_representations/sharded_graph/sharded_graph.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_generation_API/graph_generation_API.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_processing_API/nec/graph_primitives_nec.h"

#ifdef __USE_GPU__
#include "graph_processing_API/gpu/graph_primitives_gpu.cuh"
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "algorithms/label_propagation/label_propagation.h"
#include "algorithms/sssp/shortest_paths.h"
#include "algorithms/sswp/widest_paths.h"
#include "algorithms/page_rank/page_rank.h"
#include "algorithms/bfs/bfs.h"
#include "algorithms/cc/cc.h"
#include "algorithms/kcore/kcore.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "verification/shortest_paths_verification.hpp"
#include "verification/page_rank_verification.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_analytics/access_profiling.h"
#include "graph_analytics/cache_metrics.h"
#include "export_graphs/ligra_export.h"
#include "export_graphs/gapbs_export.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* graph_library_h */
