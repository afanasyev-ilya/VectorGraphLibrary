/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define __USE_GPU__

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 4.0
#define VECTOR_CORE_THRESHOLD_VALUE 4.0*VECTOR_LENGTH
#define COLLECTIVE_FRONTIER_TYPE_CHANGE_THRESHOLD 0.35

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_library.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    try
    {
        // parse args
        Parser parser;
        parser.parse_args(argc, argv);

        select_device(parser.get_device_num());

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

        // compute BFS
        cout << "Computations started..." << endl;
        cout << "Doing " << parser.get_number_of_rounds() << " BFS iterations..." << endl;
        for(int i = 0; i < parser.get_number_of_rounds(); i++)
        {
            VerticesArray<int> levels(graph, SCATTER);

            int source_vertex = graph.select_random_vertex(ORIGINAL);
            cout << "selected source vertex " << source_vertex << endl;

            performance_stats.reset_timers();
            BFS::gpu_top_down(graph, levels, source_vertex);
            performance_stats.update_timer_stats();
            performance_stats.print_timers_stats();

            // check if required
            if(parser.get_check_flag())
            {
                graph.move_to_host();
                levels.move_to_host();

                VerticesArray<int> check_levels(graph, SCATTER);
                BFS::seq_top_down(graph, check_levels, source_vertex);
                verify_results(levels, check_levels, 10);

                graph.move_to_device();
                levels.move_to_device();
            }
        }

        performance_stats.print_perf(graph.get_edges_count());
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
