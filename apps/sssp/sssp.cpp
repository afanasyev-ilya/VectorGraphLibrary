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
        vgl_library_data.init(argc, argv);
        cout << "SSSP (Single Source Shortest Paths) test..." << endl;

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

        #ifdef __USE_MPI__
        vgl_library_data.allocate_exchange_buffers(graph.get_vertices_count(), sizeof(float));
        #endif

        // do calculations
        EdgesArray_Vect<float> weights(graph);
        //weights.set_all_random(MAX_WEIGHT);
        weights.set_all_constant(1.0);

        // heat run
        VerticesArray<float> distances(graph, Parser::convert_traversal_type(parser.get_traversal_direction()));
        ShortestPaths::nec_dijkstra(graph, weights, distances, 0,
                                    parser.get_algorithm_frontier_type(),
                                    parser.get_traversal_direction());

        cout << "Computations started..." << endl;
        cout << "Doing " << parser.get_number_of_rounds() << " SSSP iterations..." << endl;
        performance_stats.reset_timers();
        for(int i = 0; i < parser.get_number_of_rounds(); i++)
        {
            int source_vertex = graph.select_random_vertex(ORIGINAL);
            cout << "selected source vertex " << source_vertex << endl;
            #ifdef __USE_MPI__
            vgl_library_data.bcast(&source_vertex, 1, 0);
            #endif

            //ftrace_region_begin("sssp");
            ShortestPaths::nec_dijkstra(graph, weights, distances, source_vertex,
                                        parser.get_algorithm_frontier_type(),
                                        parser.get_traversal_direction());
            //ftrace_region_end("sssp");

            // check if required
            if(parser.get_check_flag())
            {
                VerticesArray<float> check_distances(graph, SCATTER);
                ShortestPaths::seq_dijkstra(graph, weights, check_distances, source_vertex);
                verify_results(distances, check_distances);
            }
        }

        performance_stats.update_timer_stats();
        performance_stats.print_timers_stats();
        performance_stats.print_perf(graph.get_edges_count(), parser.get_number_of_rounds());

        vgl_library_data.finalize();
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
