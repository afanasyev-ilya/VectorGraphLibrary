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
        VGL_RUNTIME::init_library(argc, argv);
        VGL_RUNTIME::info_message("PR");

        // parse args
        Parser parser;
        parser.parse_args(argc, argv);

        // prepare graph
        VGL_Graph graph(VGL_RUNTIME::select_graph_format(parser), VGL_RUNTIME::select_graph_optimizations(parser));
        VGL_RUNTIME::prepare_graph(graph, parser);

        VerticesArray<float> page_ranks(graph);
        float convergence_factor = 1.0e-4;

        // heat run
        #ifdef __USE_MPI__
        PageRank::vgl_page_rank(graph, page_ranks, convergence_factor, parser.get_number_of_rounds());
        #endif

        VGL_RUNTIME::start_measuring_stats();
        VGL_RUNTIME::report_performance(PageRank::vgl_page_rank(graph, page_ranks, convergence_factor,
                                                                parser.get_number_of_rounds()));
        VGL_RUNTIME::stop_measuring_stats(graph.get_edges_count(), parser);

        if(parser.get_check_flag())
        {
            VerticesArray<float> seq_page_ranks(graph);
            PageRank::seq_page_rank(graph, seq_page_ranks, convergence_factor, parser.get_number_of_rounds());
            verify_ranking_results(page_ranks, seq_page_ranks);
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
