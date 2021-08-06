/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 5.0
#define VECTOR_ENGINE_THRESHOLD_VALUE 2147483646
#define VECTOR_CORE_THRESHOLD_VALUE 5*VECTOR_LENGTH

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define __PRINT_API_PERFORMANCE_STATS__
#define __PRINT_SAMPLES_PERFORMANCE_STATS__

#include "graph_library.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define base_type double

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    try
    {
        VGL_RUNTIME::init_library(argc, argv);
        VGL_RUNTIME::info_message("HITS");

        // parse args
        Parser parser;
        parser.parse_args(argc, argv);

        // prepare graph
        VGL_Graph graph(VGL_RUNTIME::select_graph_format(parser));
        VGL_RUNTIME::prepare_graph(graph, parser);

        VerticesArray<base_type> auth(graph);
        VerticesArray<base_type> hub(graph);

        #ifdef __USE_MPI__
        HITS::vgl_hits(graph, auth, hub, parser.get_number_of_rounds());
        #endif

        VGL_RUNTIME::start_measuring_stats();
        HITS::vgl_hits(graph, auth, hub, parser.get_number_of_rounds());
        VGL_RUNTIME::stop_measuring_stats(graph.get_edges_count(), parser);

        if(parser.get_check_flag())
        {
            VerticesArray<base_type> check_auth(graph);
            VerticesArray<base_type> check_hub(graph);
            HITS::seq_hits(graph, check_auth, check_hub, parser.get_number_of_rounds());

            verify_results(auth, check_auth, 0);
            verify_results(hub, check_hub, 0);
        }

        VGL_RUNTIME::finalize_library();
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
