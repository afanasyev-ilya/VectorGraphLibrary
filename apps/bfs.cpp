/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 4.0
#define VECTOR_CORE_THRESHOLD_VALUE 4.0*VECTOR_LENGTH
#define COLLECTIVE_FRONTIER_TYPE_CHANGE_THRESHOLD 0.35
//#define __PRINT_SAMPLES_PERFORMANCE_STATS__
//#define __PRINT_API_PERFORMANCE_STATS__

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../graph_library.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char * argv[])
{
    try
    {
        // parse args
        AlgorithmCommandOptionsParser parser;
        parser.parse_args(argc, argv);

        // load graph
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
                throw "Error: graph file not found";
            double t2 = omp_get_wtime();
            cout << "file " << parser.get_graph_file_name() << " loaded in " << t2 - t1 << " sec" << endl;
        }

        // print size of VectCSR graph
        graph.print_size();

        #ifdef __USE_GPU__
        graph.move_to_device();
        #endif

        // compute BFS
        cout << "Computations started..." << endl;
        cout << "Doing " << parser.get_number_of_rounds() << " BFS iterations..." << endl;
        for(int i = 0; i < parser.get_number_of_rounds(); i++)
        {
            VerticesArrayNec<int> levels(graph, SCATTER); // TODO

            int source_vertex = graph.select_random_vertex(ORIGINAL);

            #ifdef __PRINT_API_PERFORMANCE_STATS__
            PerformanceStats::reset_API_performance_timers();
            #endif

            #ifdef __USE_NEC_SX_AURORA__
            BFS::nec_top_down(graph, levels, source_vertex);
            #endif

            #ifdef __USE_GPU__
            BFS::gpu_top_down(graph, device_bfs_levels, vertex_to_check);
            #endif

            #ifdef __PRINT_API_PERFORMANCE_STATS__
            PerformanceStats::print_API_performance_timers(graph.get_edges_count());
            #endif

            // check if required
            if(parser.get_check_flag())
            {
                VerticesArrayNec<int> check_levels(graph, SCATTER);
                BFS::seq_top_down(graph, check_levels, source_vertex);
                verify_results(graph, levels, check_levels);
            }
        }

        #ifdef __SAVE_PERFORMANCE_STATS_TO_FILE__
        PerformanceStats::save_performance_to_file("bfs_td", parser.get_graph_file_name(), avg_perf);
        #endif

        #ifdef __USE_GPU__
        graph.move_to_host();
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
