/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 5.0
#define VECTOR_ENGINE_THRESHOLD_VALUE VECTOR_LENGTH*MAX_SX_AURORA_THREADS*128
#define VECTOR_CORE_THRESHOLD_VALUE 16*VECTOR_LENGTH

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_library.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char * argv[])
{
    try
    {
        cout << "SSSP (Single Source Shortest Paths) test..." << endl;
        cout << "max threads: " << omp_get_max_threads() << endl;

        // parse args
        Parser parser;
        parser.parse_args(argc, argv);

        VectCSRGraph graph;
        if(parser.get_compute_mode() == GENERATE_NEW_GRAPH)
        {
            EdgesListGraph el_graph;
            int v = pow(2.0, parser.get_scale());
            if(parser.get_graph_type() == RMAT)
                GraphGenerationAPI::R_MAT(el_graph, v, v * parser.get_avg_degree(), 57, 19, 19, 5, DIRECTED_GRAPH);
            else if(parser.get_graph_type() == RANDOM_UNIFORM)
                GraphGenerationAPI::random_uniform(el_graph, v, v * parser.get_avg_degree(), DIRECTED_GRAPH);
            graph.import(el_graph);
        }
        else if(parser.get_compute_mode() == LOAD_GRAPH_FROM_FILE)
        {
            Timer tm;
            tm.start();
            if(!graph.load_from_binary_file(parser.get_graph_file_name()))
                throw "Error: graph file not found";
            tm.end();
            tm.print_time_stats("Graph load");
        }

        // print graphs stats
        graph.print_size();
        #ifndef __USE_NEC_SX_AURORA__
        graph.print_stats();
        #endif

        // do calculations
        cout << "Computations started..." << endl;
        cout << "Doing " << parser.get_number_of_rounds() << " SSSP iterations..." << endl;
        EdgesArray_Vect<float> weights(graph);
        weights.set_all_random(1.0);
        for(int i = 0; i < parser.get_number_of_rounds(); i++)
        {
            int source_vertex = graph.select_random_vertex(ORIGINAL);
            cout << "selected source vertex " << source_vertex << endl;
            VerticesArray<float> distances(graph, Parser::convert_traversal_type(parser.get_traversal_direction()));

            performance_stats.reset_timers();
            ShortestPaths::nec_dijkstra(graph, weights, distances, source_vertex,
                                        parser.get_algorithm_frontier_type(),
                                        parser.get_traversal_direction());
            performance_stats.update_timer_stats();
            performance_stats.print_timers_stats();

            // check if required
            if(parser.get_check_flag())
            {
                VerticesArray<float> check_distances(graph, SCATTER);
                ShortestPaths::seq_dijkstra(graph, weights, check_distances, source_vertex);
                verify_results(distances, check_distances);
            }
        }
        performance_stats.print_perf(graph.get_edges_count());
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
