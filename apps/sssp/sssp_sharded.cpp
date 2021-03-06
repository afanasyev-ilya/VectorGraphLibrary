/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 5.0
#define VECTOR_ENGINE_THRESHOLD_VALUE VECTOR_LENGTH * MAX_SX_AURORA_THREADS * 4096
#define VECTOR_CORE_THRESHOLD_VALUE 5*VECTOR_LENGTH

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

        Timer tm;
        tm.start();
        ShardedCSRGraph graph(USE_SCATTER_ONLY);
        EdgesListGraph el_graph;
        if(parser.get_compute_mode() == GENERATE_NEW_GRAPH)
        {
            int v = pow(2.0, parser.get_scale());
            if(parser.get_graph_type() == RMAT)
                GraphGenerationAPI::R_MAT(el_graph, v, v * parser.get_avg_degree(), 57, 19, 19, 5, DIRECTED_GRAPH);
            else if(parser.get_graph_type() == RANDOM_UNIFORM)
                GraphGenerationAPI::random_uniform(el_graph, v, v * parser.get_avg_degree(), DIRECTED_GRAPH);

            EdgesListGraph copy_graph = el_graph;
            graph.import(copy_graph);
        }
        else if(parser.get_compute_mode() == LOAD_GRAPH_FROM_FILE)
        {
            double t1 = omp_get_wtime();
            if(!graph.load_from_binary_file(parser.get_graph_file_name()))
                throw "Error: graph file not found";
            double t2 = omp_get_wtime();
            cout << "file " << parser.get_graph_file_name() << " loaded in " << t2 - t1 << " sec" << endl;
        }
        tm.end();
        tm.print_time_stats("Generate and import");
        graph.print_size();

        tm.start();
        // generate weights
        EdgesArray_EL<int> el_weights(el_graph);
        el_weights.set_all_random(MAX_WEIGHT);

        EdgesArray_Sharded<int> weights(graph);
        weights = el_weights;
        tm.end();
        tm.print_time_stats("Generate weights");

        // compute SSSP
        cout << "Computations started..." << endl;
        cout << "Doing " << parser.get_number_of_rounds() << " SSSP iterations..." << endl;
        for(int i = 0; i < parser.get_number_of_rounds(); i++)
        {
            int source_vertex = graph.select_random_vertex();
            VerticesArray<int> distances(graph, ORIGINAL);

            performance_stats.reset_timers();
            ShortestPaths::nec_dijkstra(graph, weights, distances, source_vertex);
            performance_stats.update_timer_stats();
            //performance_stats.print_timers_stats();

            // check if required
            if(parser.get_check_flag())
            {
                VerticesArray<int> el_distances(el_graph, ORIGINAL);
                ShortestPaths::nec_dijkstra(el_graph, el_weights, el_distances, source_vertex);
                verify_results(distances, el_distances);
            }
        }
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
