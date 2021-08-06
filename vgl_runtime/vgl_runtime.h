#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "helpers/parallel_primitives/primitives.h"
#include "helpers/timer/timer.h"
#include "helpers/memory_API/memory_API.h"
#include "helpers/performance_stats/performance_stats.h"
#include "helpers/library_data/library_data.h"
#include "helpers/random_generator/random_generator.h"
#include "helpers/sorter/sorter.h"
#include "helpers/cmd_parser/cmd_parser.h"
#include "helpers/misc/extensions.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_generation/graph_generation.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "vgl_datastructures/graphs/base_graph.h"
#include "vgl_datastructures/graphs/undirected_containers/undirected_graph.h"
#include "vgl_datastructures/graphs/undirected_containers/edges_list/edges_list_graph.h"
#include "vgl_datastructures/graphs/undirected_containers/vect_csr/vect_csr_graph.h"
#include "vgl_datastructures/graphs/undirected_containers/csr/csr_graph.h"
#include "vgl_datastructures/graphs/vgl_graph/vgl_graph.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class VGL_RUNTIME
{
public:
    static void init_library(int argc, char **argv);

    static void info_message(string _algo_name);

    static void prepare_graph(VGL_Graph &_graph, Parser &_parser, DirectionType _direction = DIRECTED_GRAPH);

    static GraphType select_graph_format(Parser &_parser);

    static void finalize_library();

    static void start_measuring_stats();

    static void stop_measuring_stats(long long _edges_count, Parser &_parser);
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "vgl_runtime.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

