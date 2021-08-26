/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 4.0
#define NEC_VECTOR_ENGINE_THRESHOLD_VALUE  VECTOR_LENGTH * MAX_SX_AURORA_THREADS * 128
#define VECTOR_CORE_THRESHOLD_VALUE 2*VECTOR_LENGTH

#define COLLECTIVE_FRONTIER_TYPE_CHANGE_THRESHOLD 0.35

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_library.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    try
    {
        VGL_RUNTIME::init_library(argc, argv);
        VGL_RUNTIME::info_message("BFS");

        // parse args
        Parser parser;
        parser.parse_args(argc, argv);

        // prepare graph
        VGL_Graph graph;
        VGL_RUNTIME::prepare_graph(graph, parser);

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
        VerticesArray<int> levels(graph, SCATTER);

        for(int i = 0; i < parser.get_number_of_rounds(); i++)
        {
            int source_vertex = graph.select_random_vertex(ORIGINAL);
            cout << "selected source vertex " << source_vertex << " on run № " << i << endl;
            cout << "this vertex is " << graph.reorder(source_vertex, ORIGINAL, SCATTER) << ", " << 100.0*graph.reorder(source_vertex, ORIGINAL, SCATTER)/graph.get_vertices_count() << " % pos" << endl;

            performance_stats.reset_timers();

            //if(parser.get_algorithm_bfs() == DIRECTION_OPTIMIZING_BFS_ALGORITHM)
            //    BFS::hardwired_do_bfs(graph, levels, source_vertex, vector_extension_for_bfs, buffer1, buffer2);
            //else if(parser.get_algorithm_bfs() == TOP_DOWN_BFS_ALGORITHM)
            BFS::vgl_top_down(graph, levels, source_vertex);

            performance_stats.update_timer_stats();
            performance_stats.print_timers_stats();

            /*if(parser.get_check_flag())
            {
                VerticesArray<int> check_levels(graph, SCATTER);
                BFS::seq_top_down(graph, check_levels, source_vertex);

                verify_results(levels, check_levels, 0);
            }*/
        }
        performance_stats.print_perf(graph.get_edges_count());

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
