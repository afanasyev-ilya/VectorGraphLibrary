/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 5.0
#define VECTOR_ENGINE_THRESHOLD_VALUE 2147483646
#define VECTOR_CORE_THRESHOLD_VALUE 5*VECTOR_LENGTH

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_library.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define base_type double

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    try
    {
        vgl_library_data.init(argc, argv);
        cout << "HITS test..." << endl;

        // parse args
        Parser parser;
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

        #ifdef __USE_MPI__
        vgl_library_data.allocate_exchange_buffers(graph.get_vertices_count(), sizeof(double));
        #endif

        VerticesArray<base_type> auth(graph);
        VerticesArray<base_type> hub(graph);

        #ifdef __USE_MPI__
        HITS::vgl_hits(graph, auth, hub, parser.get_number_of_rounds());
        #endif

        performance_stats.reset_timers();
        HITS::vgl_hits(graph, auth, hub, parser.get_number_of_rounds());
        performance_stats.update_timer_stats();
        performance_stats.print_timers_stats();

        if(parser.get_check_flag())
        {
            VerticesArray<base_type> check_auth(graph);
            VerticesArray<base_type> check_hub(graph);
            HITS::seq_hits(graph, check_auth, check_hub, parser.get_number_of_rounds());

            verify_results(auth, check_auth, 0);
            verify_results(hub, check_hub, 0);
        }

        performance_stats.print_perf(graph.get_edges_count());

        vgl_library_data.finalize();
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
