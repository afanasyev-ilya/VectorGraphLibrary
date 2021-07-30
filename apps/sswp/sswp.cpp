/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 5.0
#define VECTOR_ENGINE_THRESHOLD_VALUE VECTOR_LENGTH*MAX_SX_AURORA_THREADS*128
#define VECTOR_CORE_THRESHOLD_VALUE 3*VECTOR_LENGTH

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_library.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    try
    {
        vgl_library_data.init(argc, argv);
        cout << "SSSP (Single Source Widest Paths) test..." << endl;

        // parse args
        Parser parser;
        parser.parse_args(argc, argv);

        VGL_Graph graph(VECTOR_CSR_GRAPH);

        // obtain_graph(main_graph, parser); TODO

        if(parser.get_compute_mode() == GENERATE_NEW_GRAPH)
        {
            EdgesContainer edges_container;
            int v = pow(2.0, parser.get_scale());
            if(parser.get_graph_type() == RMAT)
                GraphGenerationAPI::R_MAT(edges_container, v, v * parser.get_avg_degree(), 57, 19, 19, 5, DIRECTED_GRAPH);
            else if(parser.get_graph_type() == RANDOM_UNIFORM)
                GraphGenerationAPI::random_uniform(edges_container, v, v * parser.get_avg_degree(), DIRECTED_GRAPH);
            graph.import(edges_container);
        }
        else if(parser.get_compute_mode() == LOAD_GRAPH_FROM_FILE)
        {
            /*Timer tm;
            tm.start();
            if(!main_graph.load_from_binary_file(parser.get_graph_file_name()))
                throw "Error: graph file not found";
            tm.end();
            tm.print_time_stats("Graph load");*/
        }

        graph.print();

        // do calculations
        cout << "Computations started..." << endl;
        cout << "Doing " << parser.get_number_of_rounds() << " SSSP iterations..." << endl;

        // define BFS-levels for each graph vertex
        VerticesArray<int> levels(graph, SCATTER);
        GraphAbstractionsNEC graph_API(graph, SCATTER);


        /*EdgesArray_Vect<float> capacities(graph);
        //capacities.set_all_random(MAX_WEIGHT);
        capacities.set_all_constant(1.0);
        for(int i = 0; i < parser.get_number_of_rounds(); i++)
        {
            int source_vertex = graph.select_random_vertex(ORIGINAL);
            VerticesArray<float> widths(graph, SCATTER);

            performance_stats.reset_timers();
            SSWP::vgl_dijkstra(graph, capacities, widths, source_vertex);
            performance_stats.update_timer_stats();
            performance_stats.print_timers_stats();

            // check if required
            if(parser.get_check_flag())
            {
                VerticesArray<float> check_widths(graph, SCATTER);
                SSWP::seq_dijkstra(graph, capacities, check_widths, source_vertex);
                verify_results(widths, check_widths, 20);
            }
        }
        performance_stats.print_perf(graph.get_edges_count());*/
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
