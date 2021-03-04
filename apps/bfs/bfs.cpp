/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define __USE_NEC_SX_AURORA__

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 4.0
#define NEC_VECTOR_ENGINE_THRESHOLD_VALUE  VECTOR_LENGTH * MAX_SX_AURORA_THREADS * 128
//#define VECTOR_CORE_THRESHOLD_VALUE VECTOR_LENGTH
#define VECTOR_CORE_THRESHOLD_VALUE 2*VECTOR_LENGTH

#define COLLECTIVE_FRONTIER_TYPE_CHANGE_THRESHOLD 0.35

/*#define FRONTIER_TYPE_CHANGE_THRESHOLD 1.0
#define VE_FRONTIER_TYPE_CHANGE_THRESHOLD 1.0
#define VC_FRONTIER_TYPE_CHANGE_THRESHOLD 1.0
#define COLLECTIVE_FRONTIER_TYPE_CHANGE_THRESHOLD 1.0*/

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//#define __PRINT_API_PERFORMANCE_STATS__

#include "graph_library.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char * argv[])
{
    try
    {
        DirectionType direction = UNDIRECTED_GRAPH;

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
                GraphGenerationAPI::R_MAT(el_graph, v, v * parser.get_avg_degree(), 57, 19, 19, 5, UNDIRECTED_GRAPH);
            else if(parser.get_graph_type() == RANDOM_UNIFORM)
                GraphGenerationAPI::random_uniform(el_graph, v, v * parser.get_avg_degree(), UNDIRECTED_GRAPH);
            graph.import(el_graph);
            direction = UNDIRECTED_GRAPH;
        }
        else if(parser.get_compute_mode() == LOAD_GRAPH_FROM_FILE)
        {
            Timer tm;
            tm.start();
            if(!graph.load_from_binary_file(parser.get_graph_file_name()))
                throw "Error: graph file not found";
            tm.end();
            tm.print_time_stats("Graph load");
            direction = UNDIRECTED_GRAPH;
        }

        // print graphs stats
        graph.print_stats();

        // init ve and tmp datastructures
        BFS_GraphVE vector_extension_for_bfs(graph);
        int *buffer1, *buffer2;
        MemoryAPI::allocate_array(&buffer1, graph.get_vertices_count());
        MemoryAPI::allocate_array(&buffer2, graph.get_vertices_count());

        // compute BFS
        cout << "Computations started..." << endl;
        cout << "Doing " << parser.get_number_of_rounds() << " BFS iterations..." << endl;

        // do runs
        double avg_time = 0;
        VerticesArray<int> levels(graph, SCATTER, USE_CACHED_MODE);

        for(int i = 0; i < parser.get_number_of_rounds(); i++)
        {
            int source_vertex = graph.select_random_vertex(ORIGINAL);
            cout << "selected source vertex " << source_vertex << " on run № " << i << endl;
            cout << "this vertex is " << graph.reorder(source_vertex, ORIGINAL, SCATTER) << ", " << 100.0*graph.reorder(source_vertex, ORIGINAL, SCATTER)/graph.get_vertices_count() << " % pos" << endl;

            Timer tm;
            tm.start();
            //performance_stats.reset_timers();
            BFS::hardwired_do_bfs(graph, levels, source_vertex, vector_extension_for_bfs, buffer1, buffer2);
            //performance_stats.print_timers_stats();
            tm.end();
            avg_time += tm.get_time()/(parser.get_number_of_rounds());
            cout << "Performance of run: " << graph.get_edges_count()/(tm.get_time() * 1e6) << " MTEPS" << endl << endl;

            // check if required
            if(parser.get_check_flag())
            {
                VerticesArray<int> check_levels(graph, SCATTER);
                BFS::seq_top_down(graph, check_levels, source_vertex);

                verify_results(levels, check_levels, 0);
            }
        }
        cout << "Average performance: " << graph.get_edges_count()/(avg_time * 1e6) << " MTEPS" << endl;

        MemoryAPI::free_array(buffer1);
        MemoryAPI::free_array(buffer2);
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
