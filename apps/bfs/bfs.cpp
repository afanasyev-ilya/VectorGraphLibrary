/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 4.0
#define NEC_VECTOR_ENGINE_THRESHOLD_VALUE  VECTOR_LENGTH * MAX_SX_AURORA_THREADS * 128
#define VECTOR_CORE_THRESHOLD_VALUE 2*VECTOR_LENGTH

#define COLLECTIVE_FRONTIER_TYPE_CHANGE_THRESHOLD 0.35

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//#define __PRINT_API_PERFORMANCE_STATS__
#include "graph_library.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    try
    {
        VGL_RUNTIME::init_library(argc, argv);
        VGL_RUNTIME::info_message("BFS");

        // parse args
        Parser parser;
        parser.parse_args(argc, argv);

        // prepare graph
        VGL_Graph graph(VGL_RUNTIME::select_graph_format(parser));
        VGL_RUNTIME::prepare_graph(graph, parser);

        graph.print();

        // start algorithm
        int source_vertex = 0;
        VGL_RUNTIME::start_measuring_stats();
        VerticesArray<int> levels(graph, SCATTER);
        for(int i = 0; i < parser.get_number_of_rounds(); i++)
        {
            source_vertex = graph.select_random_nz_vertex(SCATTER);
            BFS::vgl_top_down(graph, levels, source_vertex);
        }
        VGL_RUNTIME::stop_measuring_stats(graph.get_edges_count(), parser);

        if(parser.get_check_flag())
        {
            VerticesArray<int> check_levels(graph, SCATTER);
            BFS::seq_top_down(graph, check_levels, source_vertex);
            verify_results(levels, check_levels, 0);
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
