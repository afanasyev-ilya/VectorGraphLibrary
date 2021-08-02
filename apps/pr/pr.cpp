/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 5.0
#define VECTOR_ENGINE_THRESHOLD_VALUE 2147483646
#define VECTOR_CORE_THRESHOLD_VALUE 5*VECTOR_LENGTH

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_library.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    try
    {
        VGL_COMMON_API::init_library(argc, argv);
        VGL_COMMON_API::info_message("PR");

        // parse args
        Parser parser;
        parser.parse_args(argc, argv);

        // prepare graph
        VGL_Graph graph(VECTOR_CSR_GRAPH);
        VGL_COMMON_API::prepare_graph(graph, parser);

        VerticesArray<float> page_ranks(graph);
        float convergence_factor = 1.0e-4;

        // heat run
        #ifdef __USE_MPI__
        PageRank::vgl_page_rank(graph, page_ranks, convergence_factor, parser.get_number_of_rounds());
        #endif

        VGL_COMMON_API::start_measuring_stats();
        PageRank::vgl_page_rank(graph, page_ranks, convergence_factor, parser.get_number_of_rounds());
        VGL_COMMON_API::stop_measuring_stats(graph.get_edges_count(), parser);

        if(parser.get_check_flag())
        {
            VerticesArray<float> seq_page_ranks(graph);
            PageRank::seq_page_rank(graph, seq_page_ranks, convergence_factor, parser.get_number_of_rounds());
            verify_results(page_ranks, seq_page_ranks);
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
