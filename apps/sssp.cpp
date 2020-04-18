/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 5.0

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

        //VectorisedCSRGraph<int, float> graph;
        ExtendedCSRGraph<int, float> graph;
        EdgesListGraph<int, float> rand_graph;
        if(parser.get_compute_mode() == GENERATE_NEW_GRAPH)
        {
            int vertices_count = pow(2.0, parser.get_scale());
            long long edges_count = vertices_count * parser.get_avg_degree();
            //GraphGenerationAPI<int, float>::random_uniform(rand_graph, vertices_count, edges_count, UNDIRECTED_GRAPH);
            GraphGenerationAPI<int, float>::R_MAT(rand_graph, vertices_count, edges_count, 57, 19, 19, 5, UNDIRECTED_GRAPH);
            graph.import_graph(rand_graph, VERTICES_SORTED, EDGES_SORTED, VECTOR_LENGTH, PULL_TRAVERSAL);
        }
        else if(parser.get_compute_mode() == LOAD_GRAPH_FROM_FILE)
        {
            double t1 = omp_get_wtime();
            if(!graph.load_from_binary_file(parser.get_graph_file_name()))
                throw "ERROR: graph file not found";
            double t2 = omp_get_wtime();
            cout << "file " << parser.get_graph_file_name() << " loaded in " << t2 - t1 << " sec" << endl;
        }

        GraphAnalytics graph_analytics;
        graph_analytics.analyse_graph_stats(graph, parser.get_graph_file_name());
        
        // compute SSSP
        int last_src_vertex = 0;
        cout << "Computations started..." << endl;
        ShortestPaths<int, float> sssp_operation(graph);
        float *distances;
        sssp_operation.allocate_result_memory(graph.get_vertices_count(), &distances);
        
        #ifdef __USE_GPU__
        graph.move_to_device();
        #endif

        cout << "Doing " << parser.get_number_of_rounds() << " SSSP iterations..." << endl;
        double t1 = omp_get_wtime();
        for(int i = 0; i < parser.get_number_of_rounds(); i++)
        {
            last_src_vertex = i * 100 + 1;

            #ifdef __USE_NEC_SX_AURORA__
            #ifdef __PRINT_API_PERFORMANCE_STATS__
            reset_nec_debug_timers();
            #endif
            sssp_operation.nec_dijkstra(graph, distances, last_src_vertex, parser.get_algorithm_frontier_type(),
                                        parser.get_traversal_direction());
            #ifdef __PRINT_API_PERFORMANCE_STATS__
            print_nec_debug_timers(graph);
            #endif
            #endif
            
            #ifdef __USE_GPU__
            sssp_operation.gpu_dijkstra(graph, distances, last_src_vertex, parser.get_algorithm_frontier_type(),
                                        parser.get_traversal_direction());
            #endif
        }
        double t2 = omp_get_wtime();
        
        #ifdef __USE_GPU__
        graph.move_to_host();
        #endif

        double avg_perf = parser.get_number_of_rounds() * (((double)graph.get_edges_count()) / ((t2 - t1) * 1e6));
        
        cout << "SSSP wall time: " << 1000.0 * (t2 - t1) << " ms" << endl;
        cout << "SSSP average performance: " << avg_perf << " MFLOPS" << endl << endl;

        #ifdef __SAVE_PERFORMANCE_STATS_TO_FILE__
        PerformanceStats::save_performance_to_file("sssp", parser.get_graph_file_name(), avg_perf);
        #endif

        // check if required
        if(parser.get_check_flag())
        {
            float *check_distances;
            sssp_operation.allocate_result_memory(graph.get_vertices_count(), &check_distances);
            sssp_operation.seq_dijkstra(graph, check_distances, last_src_vertex);
            
            verify_results(distances, check_distances, graph.get_vertices_count());

            sssp_operation.free_result_memory(check_distances);
        }

        sssp_operation.free_result_memory(distances);
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