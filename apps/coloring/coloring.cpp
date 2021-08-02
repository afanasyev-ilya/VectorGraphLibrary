/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 5.0
#define VECTOR_ENGINE_THRESHOLD_VALUE VECTOR_LENGTH*MAX_SX_AURORA_THREADS*32
#define VECTOR_CORE_THRESHOLD_VALUE 5*VECTOR_LENGTH

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//#define __PRINT_API_PERFORMANCE_STATS__

#include "graph_library.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    try
    {
        VGL_COMMON_API::init_library(argc, argv);
        VGL_COMMON_API::info_message("Coloring");

        // parse args
        Parser parser;
        parser.parse_args(argc, argv);

        // prepare graph
        VGL_Graph graph(VECTOR_CSR_GRAPH);
        VGL_COMMON_API::prepare_graph(graph, parser, UNDIRECTED_GRAPH);

        VerticesArray<int> colors(graph);
        VGL_COMMON_API::start_measuring_stats();
        Coloring::vgl_coloring(graph, colors);
        VGL_COMMON_API::stop_measuring_stats(graph.get_edges_count(), parser);

        if(parser.get_check_flag())
            verify_colors(graph, colors);

        VGL_COMMON_API::finalize_library();
    }
    catch (string error)
    {
        cout << error << endl;
    }
    catch (const char * error)
    {
        cout << error << endl;
    }
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
