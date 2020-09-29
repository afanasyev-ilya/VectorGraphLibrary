/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 4.0
#define VECTOR_ENGINE_THRESHOLD_VALUE VECTOR_LENGTH * MAX_SX_AURORA_THREADS * 4096
#define VECTOR_CORE_THRESHOLD_VALUE 5*VECTOR_LENGTH
#define __PRINT_SAMPLES_PERFORMANCE_STATS__

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../graph_library.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char * argv[])
{
    try
    {
        cout << "PR (connected components) test..." << endl;

        // parse args
        AlgorithmCommandOptionsParser parser;
        parser.parse_args(argc, argv);

        // load graph
        ExtendedCSRGraph<int, float> graph;
        EdgesListGraph<int, float> rand_graph;
        if(parser.get_compute_mode() == GENERATE_NEW_GRAPH)
        {
            int vertices_count = pow(2.0, parser.get_scale());
            long long edges_count = vertices_count * parser.get_avg_degree();
            GraphGenerationAPI<int, float>::R_MAT(rand_graph, vertices_count, edges_count, 57, 19, 19, 5, UNDIRECTED_GRAPH);
            graph.import_graph(rand_graph);
        }
        else if(parser.get_compute_mode() == LOAD_GRAPH_FROM_FILE)
        {
            double t1 = omp_get_wtime();
            if(!graph.load_from_binary_file(parser.get_graph_file_name()))
                throw "ERROR: graph file not found";
            double t2 = omp_get_wtime();
            cout << "file " << parser.get_graph_file_name() << " loaded in " << t2 - t1 << " sec" << endl;
        }

        PageRank<int, float> pr_operation;
        int iterations_count = parser.get_number_of_rounds();

        float *page_ranks;

        pr_operation.allocate_result_memory(graph.get_vertices_count(), &page_ranks);

        #ifdef __PRINT_API_PERFORMANCE_STATS__
        PerformanceStats::reset_API_performance_timers();
        #endif

        #ifdef __USE_NEC_SX_AURORA__
        pr_operation.nec_page_rank(graph, page_ranks, 1.0e-4, iterations_count);
        #endif

        #ifdef __USE_GPU__
        graph.move_to_device();
        pr_operation.gpu_page_rank(graph, page_ranks, 1.0e-4, iterations_count, parser.get_traversal_direction());
        graph.move_to_host();
        #endif

        #ifdef __PRINT_API_PERFORMANCE_STATS__
        PerformanceStats::print_API_performance_timers(graph.get_edges_count());
        #endif

        #ifdef __SAVE_PERFORMANCE_STATS_TO_FILE__
        PerformanceStats::save_performance_to_file("pr", parser.get_graph_file_name(), pr_operation.get_performance_per_iteration());
        #endif

        if(parser.get_check_flag())
        {
            float *check_page_ranks;
            pr_operation.allocate_result_memory(graph.get_vertices_count(), &check_page_ranks);
            //pr_operation.seq_page_rank(graph, check_page_ranks);
            pr_operation.seq_page_rank(graph, check_page_ranks, 1.0e-4, iterations_count);

            verify_results(page_ranks, check_page_ranks, graph.get_vertices_count());

            pr_operation.free_result_memory(check_page_ranks);
        }

        pr_operation.free_result_memory(page_ranks);
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
