/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 5.0
#define VECTOR_ENGINE_THRESHOLD_VALUE VECTOR_LENGTH*MAX_SX_AURORA_THREADS*128
#define VECTOR_CORE_THRESHOLD_VALUE 3*VECTOR_LENGTH

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_library.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    try
    {
        VGL_RUNTIME::init_library(argc, argv);
        VGL_RUNTIME::info_message("Maximum Flow (MF)");

        // parse args
        Parser parser;
        parser.parse_args(argc, argv);

        // prepare graph
        VGL_Graph graph(VGL_RUNTIME::select_graph_format(parser), VGL_RUNTIME::select_graph_optimizations(parser));
        VGL_RUNTIME::prepare_graph(graph, parser);

        // start algorithm
        // start algorithm
        VGL_RUNTIME::start_measuring_stats();
        double avg_perf = 0;
        for(int i = 0; i < parser.get_number_of_rounds(); i++)
        {
            int max_flow_val = 0;
            int source = graph.select_random_nz_vertex(Parser::convert_traversal_type(parser.get_traversal_direction()));
            int sink = graph.select_random_nz_vertex(Parser::convert_traversal_type(parser.get_traversal_direction()));

            avg_perf += MF::seq_ford_fulkerson(graph, source, sink, max_flow_val) / parser.get_number_of_rounds();

            if(parser.get_check_flag())
            {
                cout << "Resulting max flow: " << max_flow_val << endl;
            }
        }
        VGL_RUNTIME::stop_measuring_stats(graph.get_edges_count(), parser);
        VGL_RUNTIME::report_performance(avg_perf);

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
