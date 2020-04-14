#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <omp.h>
#include <string>
#include <iostream>
#include <vector>

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "architectures.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "common/memory_API/memory_API.h"
#include "common/random_generator/random_generation_API.h"
#include "common/sorting/sorting.h"
#include "common/verify_results/verify_results.h"
#include "common/cmd_parser/cmd_parser.h"
#include "common/parallel_primitives/copy_if.h"
#include "common/parallel_primitives/get_elements_count.h"
#include "common/performance_data/performance_data.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_representations/base_graph.h"
#include "graph_representations/edges_list_graph/edges_list_graph.h"
#include "graph_representations/extended_CSR_graph/extended_CSR_graph.h"
#include "graph_representations/sharded_graph/sharded_graph.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_generation_API/graph_generation_API.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_processing_API/nec/graph_primitives/graph_primitives_nec.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "algorithms/sssp/shortest_paths.h"
#include "algorithms/bfs/bfs.h"
#include "algorithms/cc/cc.h"
#include "algorithms/pr/pr.h"
#include "algorithms/lp/lp.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "verification/shortest_paths_verification.hpp"
#include "verification/page_rank_verification.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "export_graphs/ligra_export.h"
#include "export_graphs/gapbs_export.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_analytics/graph_analytics.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


