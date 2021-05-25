/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 5.0
#define VECTOR_ENGINE_THRESHOLD_VALUE VECTOR_LENGTH*MAX_SX_AURORA_THREADS*128

#ifdef __USE_NEC_SX_AURORA__
#define VECTOR_CORE_THRESHOLD_VALUE 5*VECTOR_LENGTH
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
        vgl_library_data.init(argc, argv);

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

        #ifdef __USE_MPI__
        vgl_library_data.allocate_exchange_buffers(graph.get_vertices_count(), sizeof(float));
        vgl_library_data.set_data_exchange_policy(RECENTLY_CHANGED);
        #endif

        EdgesArray_Vect<float> weights(graph);
        weights.set_all_constant(1.0);

        VerticesArray<float> distances(graph);

        int source_vertex = graph.select_random_vertex(ORIGINAL);
        auto min_id = [](int _a, int _b)->int
        {
            return vect_min(_a, _b);
        };
        vgl_library_data.exchange_data(&source_vertex, 1, min_id);
        cout << "source vertex: " << source_vertex << endl;

        performance_stats.reset_timers();
        SSSP::nec_dijkstra(graph, weights, distances, source_vertex, ALL_ACTIVE, parser.get_traversal_direction());
        performance_stats.update_timer_stats();
        performance_stats.print_timers_stats();

        if(parser.get_check_flag())
        {
            VerticesArray<float> check_distances(graph, SCATTER);
            ShortestPaths::seq_dijkstra(graph, weights, check_distances, source_vertex);
            verify_results(distances, check_distances);
        }

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
