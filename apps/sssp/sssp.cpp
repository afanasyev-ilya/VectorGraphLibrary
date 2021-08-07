/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 5.0
#define VECTOR_ENGINE_THRESHOLD_VALUE VECTOR_LENGTH*MAX_SX_AURORA_THREADS*128

#ifdef __USE_NEC_SX_AURORA__
#define VECTOR_CORE_THRESHOLD_VALUE 3*VECTOR_LENGTH
#endif

#ifdef __USE_MULTICORE__
#define VECTOR_CORE_THRESHOLD_VALUE 5*VECTOR_LENGTH
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_library.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    try
    {
        VGL_RUNTIME::init_library(argc, argv);
        VGL_RUNTIME::info_message("SSSP");

        // parse args
        Parser parser;
        parser.parse_args(argc, argv);

        // prepare graph
        VGL_Graph graph(VGL_RUNTIME::select_graph_format(parser));
        VGL_RUNTIME::prepare_graph(graph, parser);

        graph.print();

        // prepare weights and distances
        int source_vertex = 0;
        VerticesArray<float> distances(graph, Parser::convert_traversal_type(parser.get_traversal_direction()));
        EdgesArray<float> weights(graph);
        weights.set_all_random(MAX_WEIGHT);
        weights.print();

        // start algorithm
        VGL_RUNTIME::start_measuring_stats();
        for(int i = 0; i < parser.get_number_of_rounds(); i++)
        {
            source_vertex = graph.select_random_nz_vertex(Parser::convert_traversal_type(parser.get_traversal_direction()));
            cout << "source vertex: " << source_vertex << endl;
            ShortestPaths::vgl_dijkstra(graph, weights, distances, source_vertex,
                                        parser.get_algorithm_frontier_type(),
                                        parser.get_traversal_direction());
        }
        VGL_RUNTIME::stop_measuring_stats(graph.get_edges_count(), parser);

        if(parser.get_check_flag())
        {
            VerticesArray<float> check_distances(graph, SCATTER);
            source_vertex = graph.reorder(source_vertex, Parser::convert_traversal_type(parser.get_traversal_direction()), SCATTER);
            cout << "source vertex seq: " << source_vertex << endl;
            ShortestPaths::seq_dijkstra(graph, weights, check_distances, source_vertex);

            distances.print();
            check_distances.print();

            verify_results(distances, check_distances, graph.get_vertices_count());
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
