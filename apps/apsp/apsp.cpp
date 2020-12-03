/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 5.0

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_library.h"
#include <iostream>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char * argv[])
{
    try
    {
        cout << "APSP (All-pair Shortest Paths) test..." << endl;

        // parse args
        Parser parser;
        parser.parse_args(argc, argv);

        //VectorisedCSRGraph<int, float> graph;
        UndirectedCSRGraph<int, float> graph;
        EdgesListGraph<int, float> rand_graph;
        if(parser.get_compute_mode() == GENERATE_NEW_GRAPH)
        {
            int vertices_count = pow(2.0, parser.get_scale());
            long long edges_count = vertices_count * parser.get_avg_degree();
            //GraphGenerationAPI<int, float>::random_uniform(rand_graph, vertices_count, edges_count, UNDIRECTED_GRAPH);
            GraphGenerationAPI<int, float>::R_MAT(rand_graph, vertices_count, edges_count, 57, 19, 19, 5, UNDIRECTED_GRAPH);
            graph.import(rand_graph, VERTICES_SORTED, EDGES_SORTED, VECTOR_LENGTH, PULL_TRAVERSAL);
        }
        else if(parser.get_compute_mode() == LOAD_GRAPH_FROM_FILE)
        {
            double t1 = omp_get_wtime();
            if(!graph.load_from_binary_file(parser.get_graph_file_name()))
                throw "Error: graph file not found";
            double t2 = omp_get_wtime();
            cout << "file " << parser.get_graph_file_name() << " loaded in " << t2 - t1 << " sec" << endl;
        }

        GraphAnalytics graph_analytics;
        graph_analytics.analyse_graph_stats(graph, parser.get_graph_file_name());

        // compute APSP
        cout << "Computations started..." << endl;
        ShortestPaths<int, float> sssp_operation(graph);
        float *distances;
        sssp_operation.allocate_result_memory(graph.get_vertices_count(), &distances);

        int vertices_count = graph.get_vertices_count();
        int vertices_per_percent = vertices_count / 100;
        double t1 = omp_get_wtime();
        for(int current_vertex = 0; current_vertex < vertices_count; current_vertex++)
        {
            #ifdef __PRINT_API_PERFORMANCE_STATS__
            PerformanceStats::reset_API_performance_timers();
            #endif

            #ifdef __USE_NEC_SX_AURORA__
            sssp_operation.nec_dijkstra(graph, distances, current_vertex, ALL_ACTIVE, PUSH_TRAVERSAL);
            #endif

            #ifdef __PRINT_API_PERFORMANCE_STATS__
            PerformanceStats::print_API_performance_timers(graph.get_edges_count());
            #endif

            if(current_vertex % vertices_per_percent == 0)
            {
                cout << ((100.0 * current_vertex) / vertices_count) << "% done!" << endl;
            }
        }
        double t2 = omp_get_wtime();

        cout << "APSP wall time: " << t2 - t1 << " sec" << endl;
        cout << "APSP average performance: " << 10.0 * (((double)graph.get_edges_count()) / ((t2 - t1) * 1e6)) << " MFLOPS" << endl << endl;

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