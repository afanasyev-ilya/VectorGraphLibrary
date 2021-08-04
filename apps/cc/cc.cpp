/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 5.0
#define VECTOR_ENGINE_THRESHOLD_VALUE VECTOR_LENGTH*MAX_SX_AURORA_THREADS*128
#define VECTOR_CORE_THRESHOLD_VALUE 5*VECTOR_LENGTH

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_library.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    try
    {
        VGL_COMMON_API::init_library(argc, argv);
        VGL_COMMON_API::info_message("CC");

        // parse args
        Parser parser;
        parser.parse_args(argc, argv);

        // prepare graph
        VGL_Graph graph(VGL_COMMON_API::select_graph_format(parser));
        VGL_COMMON_API::prepare_graph(graph, parser, UNDIRECTED_GRAPH);

        VerticesArray<int> components(graph, SCATTER);

        VGL_COMMON_API::start_measuring_stats();
        if(parser.get_algorithm_cc() == SHILOACH_VISHKIN_ALGORITHM)
            ConnectedComponents::vgl_shiloach_vishkin(graph, components);
        else if(parser.get_algorithm_cc() == BFS_BASED_ALGORITHM)
            ConnectedComponents::vgl_bfs_based(graph, components);
        VGL_COMMON_API::stop_measuring_stats(graph.get_edges_count(), parser);

        if(parser.get_check_flag())
        {
            VerticesArray<int> check_components(graph, SCATTER);
            ConnectedComponents::seq_bfs_based(graph, check_components);
            equal_components(components, check_components);
        }

        VGL_COMMON_API::finalize_library();
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
