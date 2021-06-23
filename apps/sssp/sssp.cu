/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define __USE_GPU__

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 5.0

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_library.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    try
    {
        cout << "SSSP (Single Source Shortest Paths) test..." << endl;

        // parse args
        Parser parser;
        parser.parse_args(argc, argv);

        select_device(parser.get_device_num());

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

        // print size of VectCSR graph
        graph.print_size();

        // move graph to device for better performance
        graph.move_to_device();

        cout << "Computations started..." << endl;
        cout << "Doing " << parser.get_number_of_rounds() << " SSSP iterations..." << endl;

        EdgesArray_Vect<int> weights(graph);
        capacities.set_all_constant(1.0);
        //weights.set_all_random(MAX_WEIGHT);

        for(int i = 0; i < parser.get_number_of_rounds(); i++)
        {
            int source_vertex = graph.select_random_vertex(ORIGINAL);
            VerticesArray<int> distances(graph, Parser::convert_traversal_type(parser.get_traversal_direction()));

            performance_stats.reset_timers();

            ShortestPaths::gpu_dijkstra(graph, weights, distances, source_vertex,
                                        parser.get_algorithm_frontier_type(),
                                        parser.get_traversal_direction());
            performance_stats.update_timer_stats();
            performance_stats.print_timers_stats();

            // check if required
            if(parser.get_check_flag())
            {
                graph.move_to_host();
                distances.move_to_host();
                weights.move_to_host();

                VerticesArray<int> check_distances(graph, SCATTER);
                ShortestPaths::seq_dijkstra(graph, weights, check_distances, source_vertex);
                verify_results(distances, check_distances);

                graph.move_to_device();
                distances.move_to_device();
                weights.move_to_device();
            }
        }
        performance_stats.print_perf(graph.get_edges_count(), parser.get_number_of_rounds());
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
