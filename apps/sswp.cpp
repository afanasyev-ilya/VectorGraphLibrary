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
        cout << "SSWP (Single Source Widest Paths) test..." << endl;
        
        // parse args
        AlgorithmCommandOptionsParser parser;
        parser.parse_args(argc, argv);

        ExtendedCSRGraph<int, float> graph;
        if(parser.get_compute_mode() == GENERATE_NEW_GRAPH)
        {
            EdgesListGraph<int, float> rand_graph;
            int vertices_count = pow(2.0, parser.get_scale());
            long long edges_count = vertices_count * parser.get_avg_degree();
            GraphGenerationAPI<int, float>::R_MAT(rand_graph, vertices_count, edges_count, 57, 19, 19, 5, DIRECTED_GRAPH);
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

        // compute sswp
        int last_src_vertex = 0;
        cout << "Computations started..." << endl;
        WidestPaths<int, float> sswp_operation(graph);
        float *widths;
        sswp_operation.allocate_result_memory(graph.get_vertices_count(), &widths);

        cout << "Doing " << parser.get_number_of_rounds() << " SSWP iterations..." << endl;
        double t1 = omp_get_wtime();
        for(int i = 0; i < parser.get_number_of_rounds(); i++)
        {
            last_src_vertex = i * 100 + 1;

            #ifdef __PRINT_API_PERFORMANCE_STATS__
            PerformanceStats::reset_API_performance_timers();
            #endif

            #ifdef __USE_NEC_SX_AURORA__
            sswp_operation.nec_dijkstra(graph, widths, last_src_vertex, parser.get_traversal_direction());
            #endif

            #ifdef __PRINT_API_PERFORMANCE_STATS__
            PerformanceStats::print_API_performance_timers(graph.get_edges_count());
            #endif
        }
        double t2 = omp_get_wtime();
        double avg_perf = parser.get_number_of_rounds() * (((double)graph.get_edges_count()) / ((t2 - t1) * 1e6));

        cout << "SSWP wall time: " << 1000.0 * (t2 - t1) << " ms" << endl;
        cout << "SSWP average performance: " << avg_perf << " MFLOPS" << endl << endl;

        #ifdef __SAVE_PERFORMANCE_STATS_TO_FILE__
        PerformanceStats::save_performance_to_file("sswp", parser.get_graph_file_name(), avg_perf);
        #endif

        // check if required
        if(parser.get_check_flag())
        {
            float *check_widths;
            sswp_operation.allocate_result_memory(graph.get_vertices_count(), &check_widths);
            sswp_operation.seq_dijkstra(graph, check_widths, last_src_vertex);

            verify_results(widths, check_widths, graph.get_vertices_count());

            sswp_operation.free_result_memory(check_widths);
        }

        sswp_operation.free_result_memory(widths);
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
