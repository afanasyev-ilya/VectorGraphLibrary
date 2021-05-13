/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define __USE_GPU__

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 4.0

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_library.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    try
    {
        cout << "CC (Connected Components) test..." << endl;

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

        // print size of VectCSR graph
        graph.print_size();

        // move graph to device for better performance
        graph.move_to_device();

        // do calculations
        cout << "Computations started..." << endl;
        cout << "Running CC algorithm " << parser.get_number_of_rounds() << " times..." << endl;
        for(int i = 0; i < parser.get_number_of_rounds(); i++)
        {
            VerticesArray<int> components(graph, SCATTER);
            performance_stats.reset_timers();
            ConnectedComponents::gpu_shiloach_vishkin(graph, components);
            performance_stats.update_timer_stats();
            performance_stats.print_timers_stats();

            // check correctness
            if(parser.get_check_flag())
            {
                VerticesArray<int> check_components(graph, SCATTER);
                ConnectedComponents::seq_bfs_based(graph, check_components);
                equal_components(components, check_components);
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
