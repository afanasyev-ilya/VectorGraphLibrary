/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 5.0
#define VECTOR_ENGINE_THRESHOLD_VALUE VECTOR_LENGTH*MAX_SX_AURORA_THREADS*128

#ifdef __USE_NEC_SX_AURORA__
#define VECTOR_CORE_THRESHOLD_VALUE 3*VECTOR_LENGTH
#endif

#ifdef __USE_MULTICORE__
#define VECTOR_CORE_THRESHOLD_VALUE 5*VECTOR_LENGTH
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_library.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    try
    {
        cout << "SSSP (Single Source Shortest Paths) test..." << endl;
        cout << "max threads: " << omp_get_max_threads() << endl;

        // parse args
        Parser parser;
        parser.parse_args(argc, argv);

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

        // print graphs stats
        graph.print_size();

        // do calculations
        cout << "Computations started..." << endl;
        cout << "Doing " << parser.get_number_of_rounds() << " SSSP iterations..." << endl;
        EdgesArray_Vect<float> weights(graph);
        weights.set_all_random(MAX_WEIGHT);

        // compute APSP
        cout << "Computations started..." << endl;
        VerticesArray<float> distances(graph);

        int vertices_count = graph.get_vertices_count();
        int vertices_per_percent = vertices_count / 100;
        performance_stats.reset_timers();
        for(int current_vertex = 0; current_vertex < vertices_count; current_vertex++)
        {
            ShortestPaths::nec_dijkstra(graph, weights, distances, current_vertex, ALL_ACTIVE, PUSH_TRAVERSAL);
            if(current_vertex % vertices_per_percent == 0)
            {
                cout << ((100.0 * current_vertex) / vertices_count) << "% done!" << endl;
            }
        }
        performance_stats.update_timer_stats();
        performance_stats.print_timers_stats();
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
