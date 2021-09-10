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
        VGL_RUNTIME::info_message("SSWP");

        // parse args
        Parser parser;
        parser.parse_args(argc, argv);

        // prepare graph
        VGL_Graph graph(VGL_RUNTIME::select_graph_format(parser), VGL_RUNTIME::select_graph_optimizations(parser));
        VGL_RUNTIME::prepare_graph(graph, parser);

        // prepare weights and distances
        int source_vertex = 0;
        VerticesArray<float> distances(graph, Parser::convert_traversal_type(parser.get_traversal_direction()));
        EdgesArray<float> weights(graph);
        weights.set_all_random(MAX_WEIGHT);

        // start algorithm
        double avg_perf = 0;
        VGL_RUNTIME::start_measuring_stats();
        for(int i = 0; i < parser.get_number_of_rounds(); i++)
        {
            source_vertex = graph.select_random_nz_vertex(Parser::convert_traversal_type(parser.get_traversal_direction()));
            avg_perf += SSWP::vgl_dijkstra(graph, weights, distances, source_vertex) / parser.get_number_of_rounds();
        }
        VGL_RUNTIME::stop_measuring_stats(graph.get_edges_count(), parser);
        VGL_RUNTIME::report_performance(avg_perf);

        if(parser.get_check_flag())
        {
            VerticesArray<float> check_distances(graph, SCATTER);
            source_vertex = graph.reorder(source_vertex, Parser::convert_traversal_type(parser.get_traversal_direction()), SCATTER);
            SSWP::seq_dijkstra(graph, weights, check_distances, source_vertex);

            verify_results(distances, check_distances);
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
