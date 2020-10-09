/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 5.0
#define VECTOR_ENGINE_THRESHOLD_VALUE VECTOR_LENGTH * MAX_SX_AURORA_THREADS * 4096
#define VECTOR_CORE_THRESHOLD_VALUE 5*VECTOR_LENGTH
//#define __PRINT_API_PERFORMANCE_STATS__
#define __PRINT_SAMPLES_PERFORMANCE_STATS__

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../graph_library.h"
#include <iostream>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char * argv[])
{
    try
    {
        cout << "SSSP (Single Source Shortest Paths) test..." << endl;

        // parse args
        AlgorithmCommandOptionsParser parser;
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
            graph.import_graph(el_graph);
        }
        else if(parser.get_compute_mode() == LOAD_GRAPH_FROM_FILE)
        {
            double t1 = omp_get_wtime();
            if(!graph.load_from_binary_file(parser.get_graph_file_name()))
                throw "ERROR: graph file not found";
            double t2 = omp_get_wtime();
            cout << "file " << parser.get_graph_file_name() << " loaded in " << t2 - t1 << " sec" << endl;
        }

        // add weights to graph
        EdgesArrayNec<float> weights(graph);
        weights.set_all_random(MAX_WEIGHT);

        // print graph statistics
        GraphAnalytics graph_analytics;
        graph_analytics.analyse_graph_stats(graph, parser.get_graph_file_name());

        // move graph to GPU if required
        #ifdef __USE_GPU__
        graph.move_to_device();
        #endif

        // compute SSSP
        cout << "Computations started..." << endl;
        cout << "Doing " << parser.get_number_of_rounds() << " SSSP iterations..." << endl;
        double avg_perf = 0.0;
        for(int i = 0; i < parser.get_number_of_rounds(); i++)
        {
            VerticesArrayNec<float> distances(graph, convert_traversal_type(parser.get_traversal_direction()));

            int source_vertex = rand()% (graph.get_vertices_count() / 100);

            #ifdef __PRINT_API_PERFORMANCE_STATS__
            PerformanceStats::reset_API_performance_timers();
            #endif

            #ifdef __USE_NEC_SX_AURORA__
            ShortestPaths::nec_dijkstra(graph, weights, distances, source_vertex,
                                        parser.get_algorithm_frontier_type(),
                                        parser.get_traversal_direction());
            #endif

            #ifdef __PRINT_API_PERFORMANCE_STATS__
            PerformanceStats::print_API_performance_timers(graph.get_edges_count());
            #endif

            // check if required
            if(parser.get_check_flag())
            {
                VerticesArrayNec<float> check_distances(graph, SCATTER);
                ShortestPaths::seq_dijkstra(graph, weights, check_distances, source_vertex);
                verify_results(graph, distances, check_distances);
            }
        }
        
        #ifdef __USE_GPU__
        graph.move_to_host();
        #endif

        cout << "SSSP average performance: " << int(avg_perf) << " MFLOPS" << endl << endl;

        #ifdef __SAVE_PERFORMANCE_STATS_TO_FILE__
        PerformanceStats::save_performance_to_file("sssp", parser.get_graph_file_name(), int(avg_perf));
        #endif
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
