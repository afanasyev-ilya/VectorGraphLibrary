/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 4.0
#define NEC_VECTOR_ENGINE_THRESHOLD_VALUE  VECTOR_LENGTH * MAX_SX_AURORA_THREADS * 128
#define VECTOR_CORE_THRESHOLD_VALUE VECTOR_LENGTH

#define COLLECTIVE_FRONTIER_TYPE_CHANGE_THRESHOLD 0.35

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_library.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    try
    {
        VGL_COMMON_API::init_library(argc, argv);
        VGL_COMMON_API::info_message("SCC");

        // parse args
        Parser parser;
        parser.parse_args(argc, argv);

        // prepare graph
        VGL_Graph graph(VECTOR_CSR_GRAPH);
        VGL_COMMON_API::prepare_graph(graph, parser, DIRECTED_GRAPH);

        VerticesArray<int> components(graph, SCATTER);
        VGL_COMMON_API::start_measuring_stats();
        SCC::vgl_forward_backward(graph, components);
        VGL_COMMON_API::stop_measuring_stats(graph.get_edges_count());

        if(parser.get_check_flag())
        {
            VerticesArray<int> check_components(graph, SCATTER);
            SCC::seq_tarjan(graph, check_components);
            equal_components(components, check_components);
        }

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
