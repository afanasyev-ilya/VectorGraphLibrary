/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define __USE_GPU__

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 5.0

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//#define __PRINT_API_PERFORMANCE_STATS__

#include "graph_library.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    try
    {
        cout << "PR (Page Rank) test..." << endl;

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

        graph.move_to_device();

        VerticesArray<float> page_ranks(graph);
        performance_stats.reset_timers();
        float convergence_factor = 1.0e-4;
        PageRank::gpu_page_rank(graph, page_ranks, convergence_factor, parser.get_number_of_rounds(), parser.get_traversal_direction());
        performance_stats.update_timer_stats();
        performance_stats.print_timers_stats();

        graph.move_to_host();

        if(parser.get_check_flag())
        {
            VerticesArray<float> seq_page_ranks(graph);
            PageRank::seq_page_rank(graph, seq_page_ranks);
            verify_results(page_ranks, seq_page_ranks);
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
