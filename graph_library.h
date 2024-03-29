#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "settings.h"
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

// other targets can be supported here

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "framework_types.h"
#include "vgl_runtime/vgl_runtime.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// VGL datastructures
#include "vgl_datastructures/graphs/vgl_graph/vgl_graph.h"
#include "vgl_datastructures/vertices_array/vertices_array.h"
#include "vgl_datastructures/edges_array/edges_array.h"
#include "vgl_datastructures/frontier/frontier.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// VGL computational API
#include "vgl_compute_api/common/graph_abstractions.h"
#include "architecture_independent_api.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// algorithm implementations
#include "algorithms/bfs/bfs.h"
#include "algorithms/pr/pr.h"
#include "algorithms/cc/cc.h"
#include "algorithms/hits/hits.h"
#include "algorithms/coloring/coloring.h"
#include "algorithms/scc/scc.h"
#include "algorithms/rw/random_walk.h"
#include "algorithms/sssp/shortest_paths.h"
#include "algorithms/sswp/widest_paths.h"
#include "algorithms/tc/tc.h"
#include "algorithms/mf/mf.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "vgl_runtime/vgl_reminder_api.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




