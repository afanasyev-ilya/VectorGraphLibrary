/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 5.0
#define VECTOR_ENGINE_THRESHOLD_VALUE 2147483646
#define VECTOR_CORE_THRESHOLD_VALUE 5*VECTOR_LENGTH

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_library.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define base_type double

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    try
    {
        VGL_COMMON_API::init_library(argc, argv);
        VGL_COMMON_API::info_message("HITS");

        // parse args
        Parser parser;
        parser.parse_args(argc, argv);

        // prepare graph
        VGL_Graph graph(VECTOR_CSR_GRAPH);
        VGL_COMMON_API::prepare_graph(graph, parser);

        VerticesArray<base_type> auth(graph);
        VerticesArray<base_type> hub(graph);

        #ifdef __USE_MPI__
        HITS::vgl_hits(graph, auth, hub, parser.get_number_of_rounds());
        #endif

        VGL_COMMON_API::start_measuring_stats();
        HITS::vgl_hits(graph, auth, hub, parser.get_number_of_rounds());
        VGL_COMMON_API::stop_measuring_stats(graph.get_edges_count());

        if(parser.get_check_flag())
        {
            VerticesArray<base_type> check_auth(graph);
            VerticesArray<base_type> check_hub(graph);
            HITS::seq_hits(graph, check_auth, check_hub, parser.get_number_of_rounds());

            verify_results(auth, check_auth, 0);
            verify_results(hub, check_hub, 0);
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
